"""Per-video publisher with staging and generation consistency checks."""

from __future__ import annotations

import json
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any


OUTPUT_NAMES = ("timeline.json", "video_summary.json", "validation_report.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_existing_score(pilot_dir: Path) -> float | None:
    report_path = pilot_dir / "validation_report.json"
    if not report_path.is_file():
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_hash_map(values: dict[str, Any]) -> dict[str, str]:
    return {
        str(Path(path).resolve()).casefold(): str(digest)
        for path, digest in values.items()
    }


def pilot_is_resumable(*, pilot_dir: Path, source_paths: tuple[Path, ...]) -> bool:
    """Return whether a pilot can safely be skipped by ``--resume``.

    A ready report alone is insufficient: all three published artifacts must
    exist and the source hashes must still match the generation that produced
    the pilot.  This prevents resume from silently serving stale metadata.
    """

    if any(not (pilot_dir / name).is_file() for name in OUTPUT_NAMES):
        return False
    try:
        report = json.loads((pilot_dir / "validation_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if report.get("status") != "ready" or report.get("inputs_unchanged") is not True:
        return False
    expected = report.get("input_hashes")
    if not isinstance(expected, dict) or not source_paths or any(not path.is_file() for path in source_paths):
        return False
    try:
        actual = {str(path.resolve()): _sha256_file(path) for path in source_paths}
    except OSError:
        return False
    return _normalise_hash_map(expected) == _normalise_hash_map(actual)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ready":
            return None
        return float(payload.get("quality", {}).get("overall_score"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def publish_pilot(
    *,
    timeline: dict[str, Any],
    video_summary: dict[str, Any],
    validation_report: dict[str, Any],
    pilot_dir: Path,
    force: bool = False,
) -> bool:
    """Publish exactly three files; source data is never touched.

    A failed or lower-quality candidate leaves the existing pilot unchanged.
    The report is written last and acts as the commit marker for readers.
    """

    score = float(validation_report.get("quality", {}).get("overall_score", 0.0))
    current_score = read_existing_score(pilot_dir)
    if current_score is not None and score < current_score and not force:
        return False

    generation_id = validation_report["generation_id"]
    if timeline.get("generation_id") != generation_id or video_summary.get("generation_id") != generation_id:
        raise ValueError("all candidate artifacts must share generation_id")
    staging_root = pilot_dir.parents[2] / ".staging" / f"{pilot_dir.name}-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(staging_root / "timeline.json", timeline)
        _write_json(staging_root / "video_summary.json", video_summary)
        report = dict(validation_report)
        report["status"] = "ready"
        _write_json(staging_root / "validation_report.json", report)

        pilot_dir.mkdir(parents=True, exist_ok=True)
        # Replace only derived files. Raw caption/ASR/map/keyframes are outside
        # this directory and are never modified.
        for name in OUTPUT_NAMES:
            staged = staging_root / name
            destination = pilot_dir / name
            staged.replace(destination)
        return True
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        staging_parent = staging_root.parent
        if staging_parent.is_dir() and not any(staging_parent.iterdir()):
            staging_parent.rmdir()
