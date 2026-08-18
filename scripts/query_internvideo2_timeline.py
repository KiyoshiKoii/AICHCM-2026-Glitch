"""Search a precomputed InternVideo2-CLIP-S14 timeline with text queries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTokenizerFast


def _load_text_encoder(source_root: Path, checkpoint: Path, device: torch.device):
    mobileclip_dir = (
        source_root
        / "models"
        / "backbones"
        / "internvideo2"
        / "mobileclip"
    )
    package_name = "mobileclip_official"
    package = types.ModuleType(package_name)
    package.__path__ = [str(mobileclip_dir)]
    sys.modules[package_name] = package
    module_name = f"{package_name}.text_encoder"
    spec = importlib.util.spec_from_file_location(
        module_name, mobileclip_dir / "text_encoder.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the official MobileCLIP text encoder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    config = json.loads((mobileclip_dir / "configs" / "mobileclip_b.json").read_text())
    model = module.TextTransformer(config["text_cfg"], config["embed_dim"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state.get("module", state))
    text_state = {
        key.removeprefix("text_encoder."): value
        for key, value in state.items()
        if key.startswith("text_encoder.")
    }
    incompatible = model.load_state_dict(text_state, strict=True)
    # MobileCLIP's LayerNormFp32 explicitly promotes activations to float32,
    # therefore its normalization weights must stay in float32 as well.
    model.eval().to(device=device, dtype=torch.float32)
    return model, incompatible


def _parse_queries(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for index, value in enumerate(values, start=1):
        if "=" in value:
            event_id, text = value.split("=", 1)
        else:
            event_id, text = f"E{index}", value
        parsed.append((event_id.strip(), text.strip()))
    return parsed


def _top_non_overlapping(
    scores: np.ndarray,
    timestamps: np.ndarray,
    top_k: int,
    suppression_seconds: float,
) -> list[dict]:
    selected: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if all(abs(float(timestamps[index] - timestamps[other])) >= suppression_seconds for other in selected):
            selected.append(int(index))
            if len(selected) == top_k:
                break
    return [
        {
            "rank": rank,
            "timestamp_seconds": round(float(timestamps[index]), 3),
            "score": round(float(scores[index]), 6),
            "clip_index": index,
        }
        for rank, index in enumerate(selected, start=1)
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--suppression-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queries = _parse_queries(args.query)

    index_start = time.perf_counter()
    with np.load(args.index) as data:
        embeddings = data["embeddings"].astype(np.float32)
        timestamps = data["timestamps"].astype(np.float32)
        index_metadata = json.loads(str(data["metadata"]))
    index_load_seconds = time.perf_counter() - index_start

    model_start = time.perf_counter()
    model, incompatible = _load_text_encoder(args.source_root, args.text_checkpoint, device)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.tokenizer_id, local_files_only=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_start

    search_start = time.perf_counter()
    texts = [text for _, text in queries]
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.inference_mode():
        text_features = model(tokens)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scores = text_features.cpu().numpy() @ embeddings.T
    results = []
    for query_index, (event_id, text) in enumerate(queries):
        results.append(
            {
                "event_id": event_id,
                "query": text,
                "candidates": _top_non_overlapping(
                    scores[query_index],
                    timestamps,
                    args.top_k,
                    args.suppression_seconds,
                ),
            }
        )
    search_seconds = time.perf_counter() - search_start

    output = {
        "index": str(args.index),
        "video": index_metadata["video"],
        "clip_count": int(len(timestamps)),
        "device": str(device),
        "text_checkpoint": str(args.text_checkpoint),
        "timing_seconds": {
            "index_load": round(index_load_seconds, 4),
            "text_model_and_tokenizer_load": round(model_load_seconds, 4),
            "encode_queries_and_search": round(search_seconds, 4),
        },
        "load_missing_keys": list(incompatible.missing_keys),
        "load_unexpected_keys": list(incompatible.unexpected_keys),
        "results": results,
    }
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
