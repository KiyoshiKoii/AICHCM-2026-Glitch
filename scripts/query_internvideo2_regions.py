"""Find temporal regions and their leading edges in an InternVideo2 index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTokenizerFast

from query_internvideo2_timeline import _load_text_encoder


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    width = width if width % 2 else width + 1
    padding = width // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _robust_unit(values: np.ndarray) -> np.ndarray:
    low, high = np.quantile(values, [0.05, 0.95])
    if high - low < 1e-8:
        return np.full_like(values, 0.5)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _close_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = mask.copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        end = index
        while end < len(result) and not result[end]:
            end += 1
        if index > 0 and end < len(result) and end - index <= max_gap:
            result[index:end] = True
        index = end
    return result


def _runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum_length:
                runs.append((start, index - 1))
            start = None
    return runs


def _allowed_timestamps(event: dict, timestamps: np.ndarray) -> np.ndarray:
    windows = event.get("candidate_windows_seconds") or []
    if not windows:
        return np.ones(len(timestamps), dtype=bool)
    allowed = np.zeros(len(timestamps), dtype=bool)
    for window in windows:
        if not isinstance(window, list) or len(window) != 2:
            continue
        start, end = float(window[0]), float(window[1])
        allowed |= (timestamps >= start) & (timestamps <= end)
    return allowed


def _ensemble_feature(
    prompts: list[str],
    prompt_features: dict[str, np.ndarray],
) -> np.ndarray:
    feature = np.mean([prompt_features[prompt] for prompt in prompts], axis=0)
    return feature / max(np.linalg.norm(feature), 1e-12)


def _prompt_groups(event: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    """Read the predicate schema while remaining compatible with pilot specs."""

    target = event.get("target_predicates") or event.get("after_prompts") or []
    context = event.get("context_predicates") or event.get("entity_prompts") or []
    retrieval = event.get("retrieval_prompts") or [*context, *target]
    contrast = event.get("before_prompts") or []
    if not target:
        raise ValueError(f"{event.get('event_id', 'event')} has no target predicates")
    if not retrieval:
        retrieval = list(target)
    return list(retrieval), list(target), list(context), list(contrast)


def _regions_for_event(
    event: dict,
    prompt_features: dict[str, np.ndarray],
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    smooth_clips: int,
) -> dict:
    retrieval_prompts, target_prompts, context_prompts, contrast_prompts = _prompt_groups(event)
    target_feature = _ensemble_feature(target_prompts, prompt_features)
    retrieval_feature = _ensemble_feature(retrieval_prompts, prompt_features)
    target_raw = embeddings @ target_feature
    retrieval_raw = embeddings @ retrieval_feature
    target = _smooth(target_raw, smooth_clips)
    retrieval = _smooth(retrieval_raw, smooth_clips)
    target_unit = _robust_unit(target)
    retrieval_unit = _robust_unit(retrieval)
    search_score = 0.72 * target_unit + 0.28 * retrieval_unit

    if context_prompts:
        context_feature = _ensemble_feature(context_prompts, prompt_features)
        context_raw = embeddings @ context_feature
        context = _smooth(context_raw, smooth_clips)
        context_unit = _robust_unit(context)
    else:
        context_raw = np.zeros_like(target_raw)
        context_unit = np.full_like(target_raw, 0.5)

    if contrast_prompts:
        contrast_feature = _ensemble_feature(contrast_prompts, prompt_features)
        contrast_raw = embeddings @ contrast_feature
    else:
        contrast_raw = np.zeros_like(target_raw)

    high_quantile = float(event.get("high_quantile", 0.80))
    high_threshold = float(np.quantile(search_score, high_quantile))
    allowed = _allowed_timestamps(event, timestamps)
    mask = (search_score >= high_threshold) & allowed
    mask = _close_short_gaps(mask, max_gap=2)
    regions = _runs(mask, minimum_length=2)

    candidates = []
    for start, end in regions:
        peak = start + int(np.argmax(search_score[start : end + 1]))
        # Walk backward from the strict high-score region to the beginning of
        # its semantic plateau. This is the coarse leading edge to refine.
        edge_drop = float(event.get("edge_drop", 0.12))
        relaxed_threshold = max(
            float(np.quantile(search_score, 0.55)),
            float(search_score[peak] - edge_drop),
        )
        max_extension_seconds = float(event.get("max_edge_extension_seconds", 8.0))
        leading = start
        while (
            leading > 0
            and float(timestamps[start] - timestamps[leading - 1]) <= max_extension_seconds
            and allowed[leading - 1]
            and search_score[leading - 1] >= relaxed_threshold
        ):
            leading -= 1
        trailing = end
        while (
            trailing + 1 < len(search_score)
            and float(timestamps[trailing + 1] - timestamps[end]) <= max_extension_seconds
            and allowed[trailing + 1]
            and search_score[trailing + 1] >= relaxed_threshold
        ):
            trailing += 1
        selection_rule = event.get("selection_rule", "earliest_true")
        boundary = trailing if selection_rule == "latest_true" else peak if selection_rule == "representative" else leading
        lag = max(1, smooth_clips // 2)
        before_index = max(0, boundary - lag)
        after_index = min(len(target_unit) - 1, boundary + lag)
        transition_strength = float(target_unit[after_index] - target_unit[before_index])
        if selection_rule == "latest_true":
            transition_strength = -transition_strength
        # Robust units are useful for finding regions but saturate strong
        # candidates at 1.0. Rank with raw cosine scores so semantic strength
        # is not replaced by tiny transition differences.
        rank_score = float(
            target[peak]
            + 0.30 * retrieval[peak]
            + (0.10 * context_raw[peak] if context_prompts else 0.0)
        )
        if event.get("transition_required", False):
            rank_score += 0.01 * max(transition_strength, 0.0)
        candidates.append(
            {
                "boundary_seconds": round(float(timestamps[boundary]), 3),
                "leading_edge_seconds": round(float(timestamps[leading]), 3),
                "trailing_edge_seconds": round(float(timestamps[trailing]), 3),
                "strict_region_start_seconds": round(float(timestamps[start]), 3),
                "region_end_seconds": round(float(timestamps[end]), 3),
                "peak_seconds": round(float(timestamps[peak]), 3),
                "rank_score": round(rank_score, 6),
                "peak_score": round(float(search_score[peak]), 6),
                "transition_strength": round(transition_strength, 6),
                "target_score_at_boundary": round(float(target_raw[boundary]), 6),
                "target_score_at_peak": round(float(target[peak]), 6),
                "retrieval_score_at_boundary": round(float(retrieval_raw[boundary]), 6),
                "retrieval_score_at_peak": round(float(retrieval[peak]), 6),
                "context_score_at_boundary": round(float(context_raw[boundary]), 6),
                "legacy_contrast_score_at_boundary": round(float(contrast_raw[boundary]), 6),
                "boundary_clip_index": int(boundary),
                "leading_clip_index": int(leading),
                "trailing_clip_index": int(trailing),
            }
        )

    # Strict high-score runs can be split by a tiny dip even though their
    # relaxed semantic plateaus overlap. Collapse those duplicates before
    # ranking so an onset is not shifted to a later fragment of the same act.
    merged: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["leading_clip_index"]):
        combined_span = (
            candidate["trailing_edge_seconds"] - merged[-1]["leading_edge_seconds"]
            if merged
            else 0.0
        )
        if (
            not merged
            or candidate["leading_clip_index"] > merged[-1]["trailing_clip_index"]
            or combined_span > float(event.get("max_merge_span_seconds", 28.0))
        ):
            merged.append(candidate)
            continue
        previous = merged[-1]
        best = candidate if candidate["rank_score"] > previous["rank_score"] else previous
        leading_index = min(previous["leading_clip_index"], candidate["leading_clip_index"])
        trailing_index = max(previous["trailing_clip_index"], candidate["trailing_clip_index"])
        best = dict(best)
        best["leading_clip_index"] = leading_index
        best["trailing_clip_index"] = trailing_index
        best["leading_edge_seconds"] = round(float(timestamps[leading_index]), 3)
        best["trailing_edge_seconds"] = round(float(timestamps[trailing_index]), 3)
        selection_rule = event.get("selection_rule", "earliest_true")
        if selection_rule == "earliest_true":
            best["boundary_clip_index"] = leading_index
            best["boundary_seconds"] = best["leading_edge_seconds"]
        elif selection_rule == "latest_true":
            best["boundary_clip_index"] = trailing_index
            best["boundary_seconds"] = best["trailing_edge_seconds"]
        merged[-1] = best
    candidates = merged

    ranked = sorted(candidates, key=lambda item: item["rank_score"], reverse=True)
    return {
        "event_id": event["event_id"],
        "description": event.get("description", ""),
        "event_type": event.get("event_type", "action_event"),
        "selection_rule": event.get("selection_rule", "earliest_true"),
        "candidate_windows_seconds": event.get("candidate_windows_seconds", []),
        "thresholds": {
            "search_score": round(high_threshold, 6),
        },
        "regions_by_score": ranked[: int(event.get("top_k", 5))],
        "regions_chronological": sorted(candidates, key=lambda item: item["boundary_seconds"])[
            : int(event.get("top_k", 5))
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--smooth-clips", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    with np.load(args.index) as data:
        embeddings = data["embeddings"].astype(np.float32)
        timestamps = data["timestamps"].astype(np.float32)

    prompts = []
    for event in spec["events"]:
        for group in _prompt_groups(event):
            prompts.extend(group)
    prompts = list(dict.fromkeys(prompts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    model, _ = _load_text_encoder(args.source_root, args.text_checkpoint, device)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.tokenizer_id, local_files_only=True)
    load_seconds = time.perf_counter() - start

    start = time.perf_counter()
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.inference_mode():
        features = torch.nn.functional.normalize(model(tokens).float(), dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prompt_features = {
        prompt: feature
        for prompt, feature in zip(prompts, features.cpu().numpy(), strict=True)
    }
    results = [
        _regions_for_event(
            event,
            prompt_features,
            embeddings,
            timestamps,
            args.smooth_clips,
        )
        for event in spec["events"]
    ]
    search_seconds = time.perf_counter() - start

    output = {
        "index": str(args.index),
        "spec": str(args.spec),
        "smooth_clips": args.smooth_clips,
        "timing_seconds": {
            "text_model_and_tokenizer_load": round(load_seconds, 4),
            "encode_prompts_and_region_search": round(search_seconds, 4),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
