import json
import os
import sys

import numpy as np
import torch
from PIL import Image

try:
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise RuntimeError(
        "Qwen visual embedding cần sentence-transformers bản mới, "
        "transformers>=4.57.0 và qwen-vl-utils>=0.0.14."
    ) from error

from config import (
    BATCH_SIZE,
    EXTRACTION_METADATA_PATH,
    KEYFRAME_DIR,
    NPY_DIR,
    QUERY_INSTRUCTION,
    VECTOR_SIZE,
    VISUAL_ARTIFACT_ID,
    VISUAL_ATTN_IMPLEMENTATION,
    VISUAL_DTYPE,
    VISUAL_MODEL_ID,
    VISUAL_MODEL_REVISION,
    print_config,
)

# Fix encoding issue when printing Vietnamese characters in Windows Terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _resolve_torch_dtype(device: str) -> torch.dtype:
    """Resolve the requested precision without attempting FP16 on CPU."""
    if VISUAL_DTYPE == "auto":
        if device == "cuda":
            supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if supports_bf16 else torch.float16
        return torch.float32

    requested = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[VISUAL_DTYPE]
    if device == "cpu" and requested != torch.float32:
        raise RuntimeError(
            "VISUAL_DTYPE phải là float32 khi chạy CPU; "
            "hãy dùng GPU CUDA cho bfloat16/float16."
        )
    if requested == torch.bfloat16 and device == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supports_bf16:
            raise RuntimeError("GPU hiện tại không hỗ trợ bfloat16; hãy dùng VISUAL_DTYPE=float16.")
    return requested


def _resolved_model_revision(embedder: SentenceTransformer) -> str:
    """Return the pinned HF commit when Sentence Transformers exposes it."""
    try:
        first_module = embedder._first_module()
        config = getattr(getattr(first_module, "auto_model", None), "config", None)
        return getattr(config, "_commit_hash", None) or VISUAL_MODEL_REVISION
    except (AttributeError, IndexError, TypeError):
        return VISUAL_MODEL_REVISION


def _expected_metadata() -> dict[str, object]:
    return {
        "model_id": VISUAL_MODEL_ID,
        "model_revision": VISUAL_MODEL_REVISION,
        "artifact_id": VISUAL_ARTIFACT_ID,
        "vector_size": VECTOR_SIZE,
        "query_instruction": QUERY_INSTRUCTION,
    }


def _read_metadata(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as metadata_file:
        value = json.load(metadata_file)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata không hợp lệ: {path}")
    return value


def _write_metadata(path: str, metadata: dict[str, object]) -> None:
    """Atomically publish manifest state so database.py never sees half-written JSON."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2, sort_keys=True)
        metadata_file.write("\n")
    os.replace(temporary_path, path)


def _metadata_matches(metadata: dict[str, object]) -> bool:
    return all(metadata.get(key) == value for key, value in _expected_metadata().items())


def _prepare_manifest() -> None:
    """Create or validate a resumable manifest before emitting any Qwen vectors."""
    os.makedirs(NPY_DIR, exist_ok=True)

    if os.path.exists(EXTRACTION_METADATA_PATH):
        metadata = _read_metadata(EXTRACTION_METADATA_PATH)
        if not _metadata_matches(metadata):
            raise RuntimeError(
                "Artifact hiện tại không khớp model/revision/dimension/instruction. "
                "Hãy chọn VISUAL_ARTIFACT_ID mới thay vì trộn embedding khác nhau."
            )
    else:
        unknown_features = [
            name
            for name in os.listdir(NPY_DIR)
            if name.startswith("L") and name.endswith(".npy")
        ]
        if unknown_features:
            raise RuntimeError(
                "Tìm thấy .npy không có metadata trong namespace Qwen. "
                "Không thể xác nhận chúng không phải vector CLIP; hãy dùng VISUAL_ARTIFACT_ID mới."
            )

    metadata = {
        **_expected_metadata(),
        "status": "in_progress",
        "storage_dtype": "float16",
        "runtime_dtype": str(MODEL_DTYPE).replace("torch.", ""),
        "resolved_model_revision": _resolved_model_revision(model),
    }
    _write_metadata(EXTRACTION_METADATA_PATH, metadata)


def _validated_vector_count(npy_path: str, names_path: str) -> int | None:
    """Return count only when a resumed artifact is safe to reuse."""
    if not (os.path.isfile(npy_path) and os.path.isfile(names_path)):
        return None
    try:
        features = np.load(npy_path, mmap_mode="r")
        filenames = np.load(names_path, allow_pickle=False)
        if (
            features.ndim != 2
            or features.shape[1] != VECTOR_SIZE
            or not np.issubdtype(features.dtype, np.number)
            or len(filenames) != features.shape[0]
            or not np.isfinite(features).all()
        ):
            return None
        return int(features.shape[0])
    except (OSError, ValueError):
        return None


def _normalize_embeddings(embeddings: object, expected_rows: int) -> np.ndarray:
    features = np.asarray(embeddings, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    expected_shape = (expected_rows, VECTOR_SIZE)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Qwen trả vector shape {tuple(features.shape)}, cần {expected_shape}. "
            "Kiểm tra sentence-transformers có hỗ trợ truncate_dim/MRL hay không."
        )
    if not np.isfinite(features).all():
        raise RuntimeError("Qwen trả về vector chứa NaN hoặc Inf.")

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Qwen trả về vector có norm bằng 0.")
    return features / norms


def _save_artifact(npy_path: str, names_path: str, features: np.ndarray, filenames: list[str]) -> None:
    """Write feature/name pairs via temporary files to make resume safe after interruption."""
    feature_tmp_path = f"{npy_path}.tmp.npy"
    names_tmp_path = f"{names_path}.tmp.npy"
    np.save(feature_tmp_path, features.astype(np.float16, copy=False))
    np.save(names_tmp_path, np.asarray(filenames))
    os.replace(feature_tmp_path, npy_path)
    os.replace(names_tmp_path, names_path)


# ──────────────────────────────────────────────────────────────────────────────
# Khởi tạo Qwen3-VL-Embedding + auto device detection
# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = _resolve_torch_dtype(device)
model_kwargs: dict[str, object] = {"torch_dtype": MODEL_DTYPE}
if VISUAL_ATTN_IMPLEMENTATION:
    model_kwargs["attn_implementation"] = VISUAL_ATTN_IMPLEMENTATION

print(
    f"Đang tải Qwen visual embedding ({VISUAL_MODEL_ID}, "
    f"revision={VISUAL_MODEL_REVISION}) lên {device.upper()} "
    f"với {str(MODEL_DTYPE).replace('torch.', '')}..."
)
model = SentenceTransformer(
    VISUAL_MODEL_ID,
    revision=VISUAL_MODEL_REVISION,
    device=device,
    model_kwargs=model_kwargs,
)
model.eval()
print("Qwen visual embedding đã sẵn sàng.\n")


def embed_folder(folder_path: str):
    """
    Đọc tất cả ảnh trong folder_path và trả về (features, filenames).
    - features: numpy.ndarray shape (N, VECTOR_SIZE), đã L2-normalize
    - filenames: list[str] — tên file tương ứng theo thứ tự
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder không tồn tại: {folder_path}")
        return None

    image_paths = sorted([
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if os.path.splitext(filename)[1].lower() in valid_extensions
    ])

    if not image_paths:
        print(f"  ⚠️  Không tìm thấy ảnh nào trong: {folder_path}")
        return None

    print(f"  Tìm thấy {len(image_paths)} ảnh. Bắt đầu trích xuất Qwen...")

    all_features = []
    valid_filenames = []

    for index in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[index:index + BATCH_SIZE]

        # Verify trước để bỏ qua ảnh hỏng; Qwen tự đọc local path khi encode.
        valid_batch_paths = []
        for image_path in batch_paths:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                valid_batch_paths.append(image_path)
            except Exception as error:
                print(f"  ⚠️  Bỏ qua file lỗi ({os.path.basename(image_path)}): {error}")

        if not valid_batch_paths:
            continue

        image_inputs = [{"image": image_path} for image_path in valid_batch_paths]
        with torch.inference_mode():
            embeddings = model.encode(
                image_inputs,
                batch_size=len(image_inputs),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
                truncate_dim=VECTOR_SIZE,
            )
        image_features = _normalize_embeddings(embeddings, len(valid_batch_paths))
        all_features.append(image_features)
        valid_filenames.extend(os.path.basename(path) for path in valid_batch_paths)

        done = min(index + BATCH_SIZE, len(image_paths))
        print(f"  [{done}/{len(image_paths)}] batches xong...")

    if not all_features:
        return None

    final_features = np.concatenate(all_features, axis=0)
    if final_features.shape[0] != len(valid_filenames):
        raise RuntimeError("Số vector Qwen không khớp số filename hợp lệ.")
    return final_features, valid_filenames


# ──────────────────────────────────────────────────────────────────────────────
# Main — Chạy trực tiếp: python extractor.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_config()

    if not os.path.exists(KEYFRAME_DIR):
        print(f"❌ Chưa có thư mục keyframe tại: {KEYFRAME_DIR}")
        print("👉 Hãy tạo thư mục và thả data vào:")
        print(f"   {KEYFRAME_DIR}/")
        print("   ├── L25_V001/   (chứa các file .jpg/.png)")
        print("   ├── L25_V002/")
        print("   └── ...")
        sys.exit(0)

    _prepare_manifest()

    video_folders = sorted([
        directory
        for directory in os.listdir(KEYFRAME_DIR)
        if os.path.isdir(os.path.join(KEYFRAME_DIR, directory)) and directory.startswith("L")
    ])

    if not video_folders:
        print("⚠️  Không tìm thấy thư mục video nào (bắt đầu bằng 'L') trong:")
        print(f"   {KEYFRAME_DIR}")
        sys.exit(0)

    print(f"Tìm thấy {len(video_folders)} thư mục video. Bắt đầu xử lý...\n")

    feature_file_count = 0
    vector_count = 0
    for folder_name in video_folders:
        print(f"{'=' * 55}")
        print(f"  VIDEO: {folder_name}")

        npy_path = os.path.join(NPY_DIR, f"{folder_name}.npy")
        names_path = os.path.join(NPY_DIR, f"{folder_name}_filenames.npy")

        resumed_count = _validated_vector_count(npy_path, names_path)
        if resumed_count is not None:
            print("  ✅ Đã có artifact Qwen hợp lệ → Bỏ qua.")
            feature_file_count += 1
            vector_count += resumed_count
            continue

        folder_path = os.path.join(KEYFRAME_DIR, folder_name)
        result = embed_folder(folder_path)
        if result is None:
            continue

        features, filenames = result
        print(f"  Shape vector: {features.shape}")
        _save_artifact(npy_path, names_path, features, filenames)

        feature_file_count += 1
        vector_count += int(features.shape[0])
        print(f"  💾 Đã lưu: {os.path.basename(npy_path)}")
        print(f"  💾 Đã lưu: {os.path.basename(names_path)}")

    _write_metadata(
        EXTRACTION_METADATA_PATH,
        {
            **_expected_metadata(),
            "status": "complete",
            "storage_dtype": "float16",
            "runtime_dtype": str(MODEL_DTYPE).replace("torch.", ""),
            "resolved_model_revision": _resolved_model_revision(model),
            "feature_file_count": feature_file_count,
            "vector_count": vector_count,
        },
    )

    print(f"\n{'=' * 55}")
    print("✅ Hoàn tất trích xuất toàn bộ vector Qwen!")
    print(f"   Kết quả tại: {NPY_DIR}")
