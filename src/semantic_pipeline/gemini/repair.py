"""Re-extract CLIP-flagged frames and merge only those records safely.

This repair path intentionally bypasses ``global_filter_results.csv``.  It is
for frames whose metadata may have been copied from a false duplicate, so it
must ask Gemini about the actual image and leave every unflagged record intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.compact_metadata import CompactVisualRecord
from ..core.frame_deduplication import (
    DEFAULT_GLOBAL_FILTER_RESULTS_PATH,
    load_global_frame_deduplication,
)
from ..core.frame_id import frame_id_from_path, parse_frame_id
from ..core.json_io import write_json_atomically
from ..core.request_limits import is_rate_limit_error, is_transient_service_error
from ..core.visual_profiles import DEFAULT_YOUTUBE_METADATA_PATH
from .extractor import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DAILY_REQUEST_LIMIT,
    DEFAULT_MAX_INLINE_BYTES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REQUESTS_PER_MINUTE,
    MAX_RATE_LIMIT_RETRIES,
    MAX_TRANSIENT_RETRIES,
    GeminiVisualExtractor,
    caption_output_path,
    load_gemini_api_key,
    load_gemini_visual_model,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_REPAIR_RESUME_STATE_NAME = ".repair_resume.json"


@dataclass(frozen=True)
class RepairCandidate:
    frame_id: str
    claimed_score: float | None
    alternate_gap: float | None

    @property
    def video_id(self) -> str:
        return parse_frame_id(self.frame_id).video_name


def load_repair_candidates(
    report_paths: Iterable[str | Path],
    *,
    max_claimed_score: float | None = None,
    min_alternate_gap: float | None = None,
    copied_frame_ids: set[str] | None = None,
) -> list[RepairCandidate]:
    """Read unique flagged frame IDs from one or more CLIP reports.

    When either severity threshold is supplied, a frame is retained if it meets
    *either* condition. This lets callers select a high-confidence subset
    without excluding a very low-scoring caption that lacks a clear alternate.
    """
    if max_claimed_score is not None and not -1 <= max_claimed_score <= 1:
        raise ValueError("max_claimed_score must be in [-1, 1]")
    if min_alternate_gap is not None and min_alternate_gap < 0:
        raise ValueError("min_alternate_gap must not be negative")

    candidates: dict[str, RepairCandidate] = {}
    for report_path in report_paths:
        path = Path(report_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        frames = raw.get("frames") if isinstance(raw, dict) else None
        if not isinstance(frames, list):
            raise ValueError(f"CLIP report has no frames array: {path}")
        for row in frames:
            if not isinstance(row, dict) or row.get("flag") is not True:
                continue
            frame_id = row.get("frame_id")
            if not isinstance(frame_id, str):
                raise ValueError(f"Flagged report row has no frame_id: {path}")
            parse_frame_id(frame_id)
            if copied_frame_ids is not None and frame_id not in copied_frame_ids:
                continue
            claimed = _optional_float(row.get("claimed_score"))
            best = _optional_float(row.get("best_score"))
            alternate_gap = best - claimed if claimed is not None and best is not None else None
            if max_claimed_score is not None or min_alternate_gap is not None:
                passes_score = (
                    max_claimed_score is not None
                    and claimed is not None
                    and claimed <= max_claimed_score
                )
                passes_gap = (
                    min_alternate_gap is not None
                    and alternate_gap is not None
                    and alternate_gap >= min_alternate_gap
                )
                if not (passes_score or passes_gap):
                    continue
            candidates.setdefault(frame_id, RepairCandidate(frame_id, claimed, alternate_gap))
    return sorted(
        candidates.values(),
        key=lambda item: (item.video_id, parse_frame_id(item.frame_id).frame_index),
    )


def _optional_float(value: object) -> float | None:
    if not isinstance(value, (float, int)):
        return None
    return float(value)


def resolve_candidate_paths(
    candidates: Iterable[RepairCandidate], keyframe_dir: str | Path
) -> dict[str, Path]:
    """Resolve selected frame IDs from their per-video keyframe directories."""
    root = Path(keyframe_dir)
    requested_by_video: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        requested_by_video[candidate.video_id].add(candidate.frame_id)

    resolved: dict[str, Path] = {}
    for video_id, requested_ids in requested_by_video.items():
        folder = root / video_id
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                frame_id = frame_id_from_path(path)
            except ValueError:
                continue
            if frame_id in requested_ids:
                resolved[frame_id] = path
    missing = sorted(
        frame_id
        for requested_ids in requested_by_video.values()
        for frame_id in requested_ids
        if frame_id not in resolved
    )
    if missing:
        raise FileNotFoundError(f"Missing keyframe images for repair: {missing[:10]}")
    return resolved


def _load_video_records(output_dir: Path, video_id: str) -> dict[str, CompactVisualRecord]:
    path = caption_output_path(output_dir, video_id)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot repair without existing caption artifact: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Caption artifact must be a JSON array: {path}")
    records = [CompactVisualRecord.model_validate(item) for item in raw]
    return {record.frame_id: record for record in records}


def _write_video_records(
    output_dir: Path, video_id: str, records: dict[str, CompactVisualRecord]
) -> None:
    write_json_atomically(
        caption_output_path(output_dir, video_id),
        [record.model_dump(mode="json") for _, record in sorted(records.items())],
        overwrite=True,
    )


def _build_batches(
    paths_by_frame: dict[str, Path], batch_size: int, max_inline_bytes: int
) -> list[tuple[str, list[Path]]]:
    by_video: dict[str, list[Path]] = defaultdict(list)
    for frame_id, path in paths_by_frame.items():
        by_video[parse_frame_id(frame_id).video_name].append(path)
    batches: list[tuple[str, list[Path]]] = []
    for video_id, paths in sorted(by_video.items()):
        ordered = sorted(paths, key=lambda path: parse_frame_id(frame_id_from_path(path)).frame_index)
        batches.extend(
            (video_id, batch)
            for batch in GeminiVisualExtractor.batch_paths(
                ordered, batch_size=batch_size, max_inline_bytes=max_inline_bytes
            )
        )
    return batches


def default_repair_resume_state_path(output_dir: str | Path) -> Path:
    """Keep the repair checkpoint beside the caption artifacts it protects."""
    return Path(output_dir) / DEFAULT_REPAIR_RESUME_STATE_NAME


def _resume_settings(
    *,
    model_name: str,
    max_output_tokens: int,
    youtube_metadata_path: str | Path,
    use_video_context: bool,
    with_spatial: bool,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "youtube_metadata_path": str(Path(youtube_metadata_path).resolve()),
        "use_video_context": use_video_context,
        "with_spatial": with_spatial,
    }


def _resume_run_key(settings: dict[str, Any]) -> str:
    encoded = json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_resume_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": {}}
    if not path.is_file():
        raise ValueError(f"Repair resume state is not a file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Repair resume state is invalid JSON: {path}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), dict):
        raise ValueError(f"Repair resume state has an invalid format: {path}")
    return raw


def _completed_frame_ids_from_state(
    document: dict[str, Any], run_key: str
) -> set[str]:
    entry = document["runs"].get(run_key)
    if entry is None:
        return set()
    if not isinstance(entry, dict):
        raise ValueError("Repair resume state contains an invalid run entry")
    frame_ids = entry.get("completed_frame_ids", [])
    if not isinstance(frame_ids, list) or not all(isinstance(item, str) for item in frame_ids):
        raise ValueError("Repair resume state contains invalid completed_frame_ids")
    for frame_id in frame_ids:
        parse_frame_id(frame_id)
    return set(frame_ids)


def _completed_frame_ids_with_artifacts(
    output_dir: Path, frame_ids: set[str]
) -> set[str]:
    """Only skip a checkpointed frame when its repaired artifact still exists."""
    frame_ids_by_video: dict[str, set[str]] = defaultdict(set)
    for frame_id in frame_ids:
        frame_ids_by_video[parse_frame_id(frame_id).video_name].add(frame_id)

    completed: set[str] = set()
    for video_id, requested_ids in frame_ids_by_video.items():
        path = caption_output_path(output_dir, video_id)
        if not path.is_file():
            continue
        existing = _load_video_records(output_dir, video_id)
        completed.update(requested_ids.intersection(existing))
    return completed


def _write_resume_document(
    *,
    path: Path,
    document: dict[str, Any],
    run_key: str,
    settings: dict[str, Any],
    completed_frame_ids: set[str],
) -> None:
    document["runs"][run_key] = {
        "settings": settings,
        "completed_frame_ids": sorted(completed_frame_ids),
    }
    write_json_atomically(path, document, overwrite=True)


def run_repair(
    *,
    report_paths: Iterable[str | Path],
    keyframe_dir: str | Path,
    output_dir: str | Path,
    api_key: str | None,
    model_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    youtube_metadata_path: str | Path = DEFAULT_YOUTUBE_METADATA_PATH,
    use_video_context: bool = True,
    with_spatial: bool = True,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    max_concurrent_requests: int = DEFAULT_REQUESTS_PER_MINUTE,
    daily_request_limit: int = DEFAULT_DAILY_REQUEST_LIMIT,
    daily_requests_already_used: int = 0,
    request_budget_state_path: str | Path | None = None,
    max_claimed_score: float | None = None,
    min_alternate_gap: float | None = None,
    only_copied_frames: bool = False,
    global_filter_results_path: str | Path = DEFAULT_GLOBAL_FILTER_RESULTS_PATH,
    limit: int | None = None,
    resume: bool = False,
    resume_state_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repair flagged metadata, optionally resuming after completed batches."""
    if not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")
    if not 1 <= requests_per_minute or not 1 <= max_concurrent_requests <= requests_per_minute:
        raise ValueError("invalid RPM/concurrency configuration")
    model_name = model_name or load_gemini_visual_model()
    copied_frame_ids: set[str] | None = None
    if only_copied_frames:
        deduplication = load_global_frame_deduplication(global_filter_results_path)
        copied_frame_ids = {
            frame_id
            for frame_id, representative_id in deduplication.representative_by_frame.items()
            if frame_id != representative_id
        }
    candidates = load_repair_candidates(
        report_paths,
        max_claimed_score=max_claimed_score,
        min_alternate_gap=min_alternate_gap,
        copied_frame_ids=copied_frame_ids,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        candidates = candidates[:limit]

    output_root = Path(output_dir)
    state_path = (
        Path(resume_state_path)
        if resume_state_path is not None
        else default_repair_resume_state_path(output_root)
    )
    settings = _resume_settings(
        model_name=model_name,
        max_output_tokens=max_output_tokens,
        youtube_metadata_path=youtube_metadata_path,
        use_video_context=use_video_context,
        with_spatial=with_spatial,
    )
    run_key = _resume_run_key(settings)
    state_document = _load_resume_document(state_path)
    checkpointed_frame_ids = (
        _completed_frame_ids_from_state(state_document, run_key) if resume else set()
    )
    candidate_frame_ids = {candidate.frame_id for candidate in candidates}
    completed_frame_ids = _completed_frame_ids_with_artifacts(
        output_root, checkpointed_frame_ids.intersection(candidate_frame_ids)
    )
    pending_candidates = [
        candidate for candidate in candidates if candidate.frame_id not in completed_frame_ids
    ]

    paths_by_frame = resolve_candidate_paths(pending_candidates, keyframe_dir)
    batches = _build_batches(paths_by_frame, batch_size, max_inline_bytes)
    summary: dict[str, Any] = {
        "candidates": len(candidates),
        "pending_candidates": len(pending_candidates),
        "resumed_frames": len(completed_frame_ids),
        "videos": len({candidate.video_id for candidate in candidates}),
        "batches": len(batches),
        "repaired": 0,
        "api_frames": 0,
        "daily_limit_reached": False,
        "daily_quota_reported": False,
        "dry_run": dry_run,
        "only_copied_frames": only_copied_frames,
        "resume": resume,
        "resume_state_path": str(state_path),
    }
    if resume:
        print(
            f"Repair resume: skipped={len(completed_frame_ids)} "
            f"pending={len(pending_candidates)} state={state_path}"
        )
    if dry_run or not batches:
        return summary

    existing_by_video = {
        video_id: _load_video_records(output_root, video_id)
        for video_id, _ in batches
    }
    extractor = GeminiVisualExtractor(
        api_key=api_key,
        model_name=model_name,
        youtube_metadata_path=youtube_metadata_path,
        use_video_context=use_video_context,
        max_output_tokens=max_output_tokens,
    )

    next_batch_index = 0
    previous_window_started: float | None = None
    transient_retry_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    rate_retry_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    completed_batches = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        while next_batch_index < len(batches):
            if previous_window_started is not None:
                wait_seconds = 60 - (time.monotonic() - previous_window_started)
                if wait_seconds > 0:
                    print(f"RPM window complete; waiting {wait_seconds:.1f}s before next repair window.")
                    time.sleep(wait_seconds)

            window_size = min(requests_per_minute, len(batches) - next_batch_index)
            window = batches[next_batch_index : next_batch_index + window_size]
            next_batch_index += len(window)
            previous_window_started = time.monotonic()
            futures = {
                executor.submit(extractor.extract_batch, paths, with_spatial=with_spatial): (video_id, paths)
                for video_id, paths in window
            }
            retry_batches: list[tuple[str, list[Path]]] = []
            unexpected_error: Exception | None = None
            for future in as_completed(futures):
                video_id, paths = futures[future]
                try:
                    records = future.result()
                except Exception as error:
                    if is_rate_limit_error(error):
                        retry_key = (
                            video_id,
                            tuple(frame_id_from_path(path) for path in paths),
                        )
                        attempt = rate_retry_counts.get(retry_key, 0) + 1
                        if attempt <= MAX_RATE_LIMIT_RETRIES:
                            rate_retry_counts[retry_key] = attempt
                            retry_batches.append((video_id, paths))
                            print(
                                f"RPM quota reached for repair {video_id}; "
                                f"retry {attempt}/{MAX_RATE_LIMIT_RETRIES} in next window."
                            )
                        else:
                            unexpected_error = RuntimeError(
                                f"RPM quota remained unavailable after {MAX_RATE_LIMIT_RETRIES} "
                                f"retries for repair {video_id}"
                            )
                    elif is_transient_service_error(error):
                        retry_key = (
                            video_id,
                            tuple(frame_id_from_path(path) for path in paths),
                        )
                        attempt = transient_retry_counts.get(retry_key, 0) + 1
                        if attempt <= MAX_TRANSIENT_RETRIES:
                            transient_retry_counts[retry_key] = attempt
                            retry_batches.append((video_id, paths))
                            print(
                                f"Gemini service unavailable for repair {video_id}; "
                                f"retry {attempt}/{MAX_TRANSIENT_RETRIES} in next window."
                            )
                        else:
                            unexpected_error = RuntimeError(
                                f"Gemini remained unavailable after {MAX_TRANSIENT_RETRIES} "
                                f"retries for repair {video_id}"
                            )
                    else:
                        unexpected_error = error
                    continue

                expected_ids = {frame_id_from_path(path) for path in paths}
                returned_ids = {record.frame_id for record in records}
                if returned_ids != expected_ids:
                    unexpected_error = ValueError(
                        f"Repair result IDs differ from requested IDs; "
                        f"expected={sorted(expected_ids)}, returned={sorted(returned_ids)}"
                    )
                    continue
                existing = existing_by_video[video_id]
                existing.update({record.frame_id: record for record in records})
                _write_video_records(output_root, video_id, existing)
                completed_frame_ids.update(expected_ids)
                _write_resume_document(
                    path=state_path,
                    document=state_document,
                    run_key=run_key,
                    settings=settings,
                    completed_frame_ids=completed_frame_ids,
                )
                summary["repaired"] += len(records)
                summary["api_frames"] += len(records)
                completed_batches += 1
                retry_key = (
                    video_id,
                    tuple(frame_id_from_path(path) for path in paths),
                )
                transient_retry_counts.pop(retry_key, None)
                rate_retry_counts.pop(retry_key, None)
                print(
                    f"repair_batch={completed_batches}/{len(batches)} video={video_id} "
                    f"api_frames={summary['api_frames']} repaired={summary['repaired']}"
                )
            if unexpected_error is not None:
                raise unexpected_error
            if retry_batches:
                batches.extend(retry_batches)
    summary["batches"] = completed_batches
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--keyframe-dir", type=Path, default=Path("data/keyframes"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata/caption"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-inline-bytes", type=int, default=DEFAULT_MAX_INLINE_BYTES)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--youtube-metadata", type=Path, default=DEFAULT_YOUTUBE_METADATA_PATH)
    parser.add_argument("--generic-prompt", action="store_true")
    parser.add_argument("--without-spatial", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument(
        "--daily-request-limit",
        type=int,
        default=DEFAULT_DAILY_REQUEST_LIMIT,
        help="Legacy compatibility option; daily quota is checked manually and not enforced locally",
    )
    parser.add_argument(
        "--daily-requests-already-used",
        type=int,
        default=0,
        help="Legacy compatibility option; ignored because daily quota is checked manually",
    )
    parser.add_argument(
        "--request-budget-state",
        type=Path,
        help="Legacy compatibility option; no local daily budget state is written",
    )
    parser.add_argument(
        "--global-filter-results",
        type=Path,
        default=DEFAULT_GLOBAL_FILTER_RESULTS_PATH,
        help="CSV used with --only-copied-frames to identify dedup-copied records",
    )
    parser.add_argument(
        "--only-copied-frames",
        action="store_true",
        help="repair only flagged frames mapped to another representative by the global filter",
    )
    parser.add_argument("--max-claimed-score", type=float)
    parser.add_argument("--min-alternate-gap", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip frames recorded as successfully repaired by the same repair settings",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="repair checkpoint path (default: <output-dir>/.repair_resume.json)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_repair(
        report_paths=args.report,
        keyframe_dir=args.keyframe_dir,
        output_dir=args.output_dir,
        api_key=load_gemini_api_key(),
        batch_size=args.batch_size,
        max_inline_bytes=args.max_inline_bytes,
        max_output_tokens=args.max_output_tokens,
        youtube_metadata_path=args.youtube_metadata,
        use_video_context=not args.generic_prompt,
        with_spatial=not args.without_spatial,
        requests_per_minute=args.requests_per_minute,
        max_concurrent_requests=args.max_concurrent_requests,
        daily_request_limit=args.daily_request_limit,
        daily_requests_already_used=args.daily_requests_already_used,
        request_budget_state_path=args.request_budget_state,
        only_copied_frames=args.only_copied_frames,
        global_filter_results_path=args.global_filter_results,
        max_claimed_score=args.max_claimed_score,
        min_alternate_gap=args.min_alternate_gap,
        limit=args.limit,
        resume=args.resume,
        resume_state_path=args.resume_state,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
