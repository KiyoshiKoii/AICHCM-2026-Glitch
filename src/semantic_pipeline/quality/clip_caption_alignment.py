"""CLIP-based post-run verifier for Gemini caption/frame alignment.

This is a local, read-only audit.  It does not call Gemini and should be run
after a caption directory has finished generating.  Captions are compared
within configurable batches so an accidentally swapped caption is surfaced by
an alternate image scoring higher than its claimed image.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.environment import get_env_value
from ..core.frame_id import frame_id_from_path, parse_frame_id


DEFAULT_MODEL_ID = get_env_value("CAPTION_AUDIT_CLIP_MODEL") or "openai/clip-vit-base-patch32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CaptionItem:
    frame_id: str
    video_name: str
    image_path: Path
    caption: str


def alignment_rows(
    frame_ids: Sequence[str],
    scores: Sequence[Sequence[float]],
    *,
    min_score: float,
    min_margin: float,
) -> list[dict[str, Any]]:
    """Turn an image/text similarity matrix into auditable row-level flags.

    ``scores[i][j]`` is the CLIP score between image ``i`` and caption ``j``.
    The diagonal is the claimed image-caption pairing.  This function is
    deliberately dependency-free so it can be unit tested without CLIP.
    """
    if len(scores) != len(frame_ids):
        raise ValueError("score row count must equal frame ID count")
    rows: list[dict[str, Any]] = []
    for index, frame_id in enumerate(frame_ids):
        row = list(scores[index])
        if len(row) != len(frame_ids):
            raise ValueError("similarity matrix must be square")
        claimed = float(row[index])
        best_index = max(range(len(row)), key=row.__getitem__) if row else index
        best_score = float(row[best_index]) if row else claimed
        alternate_scores = [value for pos, value in enumerate(row) if pos != index]
        best_alternate = max(alternate_scores) if alternate_scores else claimed
        margin = claimed - float(best_alternate)
        reasons: list[str] = []
        diagnostics: list[str] = []
        if claimed < min_score:
            reasons.append("low_claimed_score")
        if best_index != index and best_score - claimed >= min_margin:
            reasons.append("alternate_image_higher")
        if margin < min_margin and len(row) > 1:
            # Adjacent video frames often depict the same shot. Keep this as
            # context in the report, but do not treat similarity alone as a
            # swapped-caption failure.
            diagnostics.append("low_alternate_margin")
        rows.append(
            {
                "frame_id": frame_id,
                "claimed_score": round(claimed, 6),
                "best_score": round(best_score, 6),
                "best_match_frame_id": frame_ids[best_index],
                "margin_vs_best_alternate": round(margin, 6),
                "flag": bool(reasons),
                "reasons": reasons,
                "diagnostics": diagnostics,
            }
        )
    return rows


def _read_caption_files(
    caption_dir: Path,
    video_prefix: str,
    video_id: str | None = None,
) -> list[CaptionItem]:
    items: list[CaptionItem] = []
    pattern = f"{video_id}.json" if video_id else f"{video_prefix}_V*.json"
    for path in sorted(caption_dir.glob(pattern)):
        video_name = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read caption file {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Caption file must contain a list: {path}")
        for record in payload:
            if not isinstance(record, dict) or not record.get("frame_id"):
                continue
            frame_id = str(record["frame_id"])
            parsed = parse_frame_id(frame_id)
            if not str(parsed.video_name).startswith(video_prefix):
                continue
            if video_id is not None and parsed.video_name != video_id:
                continue
            caption = str(record.get("caption") or "").strip()
            if caption:
                items.append(
                    CaptionItem(frame_id, parsed.video_name, Path(), caption)
                )
    return sorted(items, key=lambda item: (item.video_name, parse_frame_id(item.frame_id).frame_index))


def _index_images(keyframe_dir: Path, video_names: Iterable[str]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for video_name in sorted(set(video_names)):
        folder = keyframe_dir / video_name
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    indexed[frame_id_from_path(path)] = path
                except ValueError:
                    continue
    return indexed


def _load_clip(model_id: str):
    """Load CLIP lazily; importing this module never downloads model weights."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "CLIP verification requires torch, transformers, and Pillow"
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return torch, model, processor, device


def _clip_similarity(
    items: Sequence[CaptionItem],
    *,
    model_id: str,
    batch_size: int,
    min_score: float,
    min_margin: float,
):
    from PIL import Image

    torch, model, processor, device = _load_clip(model_id)
    all_rows: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        images = [Image.open(item.image_path).convert("RGB") for item in batch]
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        text_inputs = processor(
            text=[f"a photo of {item.caption}" for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
        if hasattr(image_features, "image_embeds"):
            image_features = image_features.image_embeds
        if hasattr(text_features, "text_embeds"):
            text_features = text_features.text_embeds
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        matrix = (image_features @ text_features.T).detach().cpu().tolist()
        rows = alignment_rows(
            [item.frame_id for item in batch],
            matrix,
            min_score=min_score,
            min_margin=min_margin,
        )
        for item, row in zip(batch, rows):
            row.update(
                {
                    "video_name": item.video_name,
                    "image_path": str(item.image_path),
                    "caption": item.caption,
                    "batch_index": start // batch_size + 1,
                }
            )
            all_rows.append(row)
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-prefix", default="L21")
    parser.add_argument(
        "--video-id",
        help="Audit one video only, for example L21_V001; overrides the broad prefix scan",
    )
    parser.add_argument("--caption-dir", type=Path, default=Path("data/metadata/caption"))
    parser.add_argument("--keyframe-dir", type=Path, default=Path("data/keyframes"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/verification/clip_L21_report.json"),
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any caption has no matching keyframe image",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2 for alternate matching")
    collection_dir = args.caption_dir / (
        args.video_id.split("_", maxsplit=1)[0] if args.video_id else args.video_prefix
    )
    items = _read_caption_files(collection_dir, args.video_prefix, args.video_id)
    image_index = _index_images(args.keyframe_dir, [item.video_name for item in items])
    missing = [item.frame_id for item in items if item.frame_id not in image_index]
    if missing and args.strict:
        raise SystemExit(f"Missing {len(missing)} keyframe images; first={missing[:3]}")
    items = [
        CaptionItem(item.frame_id, item.video_name, image_index[item.frame_id], item.caption)
        for item in items
        if item.frame_id in image_index
    ]
    rows = _clip_similarity(
        items,
        model_id=args.model_id,
        batch_size=args.batch_size,
        min_score=args.min_score,
        min_margin=args.min_margin,
    )
    flagged = [row for row in rows if row["flag"]]
    summary = {
        "video_prefix": args.video_prefix,
        "video_id": args.video_id,
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "frames_scored": len(rows),
        "missing_images": len(missing),
        "flagged_frames": len(flagged),
        "mean_claimed_score": round(sum(row["claimed_score"] for row in rows) / len(rows), 6)
        if rows
        else None,
        "mean_margin_vs_best_alternate": round(
            sum(row["margin_vs_best_alternate"] for row in rows) / len(rows), 6
        )
        if rows
        else None,
    }
    report = {"summary": summary, "frames": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
