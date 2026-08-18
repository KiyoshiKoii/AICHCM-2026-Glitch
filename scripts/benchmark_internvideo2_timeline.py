"""Benchmark an InternVideo2-S14 timeline index on one local video.

This script intentionally uses the official OpenGVLab source tree instead of
vendoring the model implementation.  It measures model loading, sequential
video decoding, clip inference, and index writing separately.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch


def _install_flash_attention_import_shims() -> None:
    """Allow the official model to use its non-FlashAttention code path."""

    if "flash_attn" in sys.modules:
        return

    flash_attn = types.ModuleType("flash_attn")
    flash_interface = types.ModuleType("flash_attn.flash_attn_interface")
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    modules = types.ModuleType("flash_attn.modules")
    mlp = types.ModuleType("flash_attn.modules.mlp")
    ops = types.ModuleType("flash_attn.ops")
    rms_norm = types.ModuleType("flash_attn.ops.rms_norm")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("FlashAttention shim was called unexpectedly")

    class _UnavailableModule(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            raise RuntimeError("A fused FlashAttention module was requested")

    flash_interface.flash_attn_varlen_qkvpacked_func = _unavailable
    bert_padding.unpad_input = _unavailable
    bert_padding.pad_input = _unavailable
    mlp.FusedMLP = _UnavailableModule
    rms_norm.DropoutAddRMSNorm = _UnavailableModule

    sys.modules.update(
        {
            "flash_attn": flash_attn,
            "flash_attn.flash_attn_interface": flash_interface,
            "flash_attn.bert_padding": bert_padding,
            "flash_attn.modules": modules,
            "flash_attn.modules.mlp": mlp,
            "flash_attn.ops": ops,
            "flash_attn.ops.rms_norm": rms_norm,
        }
    )


def _load_model(
    source_root: Path,
    checkpoint: Path,
    clip_checkpoint: Path,
    device: torch.device,
):
    _install_flash_attention_import_shims()
    # Import only the official visual backbone. Importing OpenGVLab's top-level
    # ``models`` package also pulls training-only dependencies such as PEFT.
    model_dir = source_root / "models" / "backbones" / "internvideo2"
    package_name = "internvideo2_official"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    module_name = f"{package_name}.internvideo2_clip_vision"
    spec = importlib.util.spec_from_file_location(
        module_name, model_dir / "internvideo2_clip_vision.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the official InternVideo2 visual backbone")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    InternVideo2 = module.InternVideo2

    model = InternVideo2(
        in_chans=3,
        patch_size=14,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.0,
        head_drop_path_rate=0.0,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        init_values=0.1,
        qk_normalization=True,
        depth=12,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        checkpoint_num=0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state.get("module", state))
    backbone_incompatible = model.load_state_dict(state, strict=False)

    clip_state = torch.load(clip_checkpoint, map_location="cpu", weights_only=True)
    clip_state = clip_state.get("model", clip_state.get("module", clip_state))
    vision_state = {
        key.removeprefix("vision_encoder."): value
        for key, value in clip_state.items()
        if key.startswith("vision_encoder.")
    }
    clip_incompatible = model.load_state_dict(vision_state, strict=False)
    vision_align = torch.nn.Sequential(
        torch.nn.LayerNorm(768),
        torch.nn.Linear(768, 512),
    )
    align_state = {
        key.removeprefix("vision_align."): value
        for key, value in clip_state.items()
        if key.startswith("vision_align.")
    }
    vision_align.load_state_dict(align_state, strict=True)
    model.eval().to(device=device, dtype=torch.float16)
    vision_align.eval().to(device=device, dtype=torch.float16)
    load_report = {
        "backbone_missing_keys": list(backbone_incompatible.missing_keys),
        "backbone_unexpected_keys": list(backbone_incompatible.unexpected_keys),
        "clip_override_key_count": len(vision_state),
        "clip_missing_key_count": len(clip_incompatible.missing_keys),
        "clip_unexpected_keys": list(clip_incompatible.unexpected_keys),
    }
    return model, vision_align, load_report


def _decode_sampled_frames(video: Path, sample_fps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0:
        raise RuntimeError(f"Invalid source FPS: {source_fps}")

    sampled: list[np.ndarray] = []
    timestamps: list[float] = []
    frame_index = 0
    sample_index = 0
    next_source_index = 0.0

    while True:
        # ``grab`` advances the decoder without materializing a BGR image for
        # every native frame. Only sampled frames pay the retrieve/convert cost.
        if not capture.grab():
            break
        if frame_index + 1e-6 >= next_source_index:
            ok, frame = capture.retrieve()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            sampled.append(frame)
            timestamps.append(frame_index / source_fps)
            sample_index += 1
            next_source_index = sample_index * source_fps / sample_fps
        frame_index += 1

    capture.release()
    if not sampled:
        raise RuntimeError("No frames were decoded")

    metadata = {
        "source_fps": source_fps,
        "source_frames_reported": source_frames,
        "source_frames_decoded": frame_index,
        "width": width,
        "height": height,
        "duration_seconds": frame_index / source_fps,
        "sample_fps": sample_fps,
        "sampled_frames": len(sampled),
    }
    return np.stack(sampled), np.asarray(timestamps, dtype=np.float32), metadata


def _encode_clips(
    model,
    vision_align,
    frames: np.ndarray,
    timestamps: np.ndarray,
    sample_fps: float,
    stride_seconds: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    clip_frames = 8
    stride_frames = max(1, round(stride_seconds * sample_fps))
    starts = np.arange(0, max(1, len(frames) - clip_frames + 1), stride_frames, dtype=np.int64)
    if starts[-1] + clip_frames > len(frames):
        starts = starts[:-1]
    if not len(starts):
        starts = np.asarray([0], dtype=np.int64)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    outputs: list[np.ndarray] = []

    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            indices = batch_starts[:, None] + np.arange(clip_frames)[None, :]
            batch = torch.from_numpy(frames[indices]).to(device=device, non_blocking=True)
            batch = batch.permute(0, 1, 4, 2, 3).to(dtype=torch.float16).div_(255.0)
            batch = batch.sub_(mean).div_(std).permute(0, 2, 1, 3, 4)
            features = vision_align(model(batch, use_image=False))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float16))

    centers = starts + (clip_frames - 1) / 2
    center_times = np.interp(centers, np.arange(len(timestamps)), timestamps).astype(np.float32)
    return np.concatenate(outputs, axis=0), center_times


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    model, vision_align, load_report = _load_model(
        args.source_root,
        args.checkpoint,
        args.clip_checkpoint,
        device,
    )
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    frames, timestamps, video_metadata = _decode_sampled_frames(args.video, args.sample_fps)
    decode_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    features, center_times = _encode_clips(
        model,
        vision_align,
        frames,
        timestamps,
        args.sample_fps,
        args.stride_seconds,
        args.batch_size,
        device,
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - stage_start

    del frames
    stage_start = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "video": str(args.video),
        "model": "OpenGVLab InternVideo2-CLIP-S14 distilled video encoder",
        "checkpoint": str(args.checkpoint),
        "clip_checkpoint": str(args.clip_checkpoint),
        "clip_frames": 8,
        "stride_seconds": args.stride_seconds,
        "batch_size": args.batch_size,
        "embedding_dim": int(features.shape[1]),
        "clip_count": int(features.shape[0]),
        "dtype": str(features.dtype),
        "device": torch.cuda.get_device_name(device),
        "video_metadata": video_metadata,
        "load_report": load_report,
    }
    np.savez(args.output, embeddings=features, timestamps=center_times, metadata=json.dumps(metadata))
    write_seconds = time.perf_counter() - stage_start
    total_seconds = time.perf_counter() - total_start

    result = {
        **metadata,
        "timing_seconds": {
            "model_load": round(model_load_seconds, 3),
            "decode_and_resize": round(decode_seconds, 3),
            "clip_inference": round(inference_seconds, 3),
            "index_write": round(write_seconds, 3),
            "total": round(total_seconds, 3),
        },
        "throughput": {
            "source_video_seconds_per_wall_second_inference": round(
                video_metadata["duration_seconds"] / inference_seconds, 3
            ),
            "clips_per_second": round(len(features) / inference_seconds, 3),
        },
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 1),
        "index_size_mb": round(args.output.stat().st_size / (1024**2), 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
