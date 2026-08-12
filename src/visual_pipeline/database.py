import json
import os
import sys
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    COLLECTION_NAME,
    EXTRACTION_METADATA_PATH,
    INDEX_METADATA_PATH,
    NPY_DIR,
    QDRANT_DB_PATH,
    QUERY_INSTRUCTION,
    VECTOR_SIZE,
    VISUAL_ARTIFACT_ID,
    VISUAL_MODEL_ID,
    VISUAL_MODEL_REVISION,
    print_config,
)

# Fix encoding cho in tiếng Việt trên Terminal Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _expected_metadata() -> dict[str, object]:
    return {
        "model_id": VISUAL_MODEL_ID,
        "model_revision": VISUAL_MODEL_REVISION,
        "artifact_id": VISUAL_ARTIFACT_ID,
        "vector_size": VECTOR_SIZE,
        "query_instruction": QUERY_INSTRUCTION,
    }


def _read_json(path: str) -> dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            value = json.load(json_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Không đọc được metadata: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Metadata không hợp lệ: {path}")
    return value


def _write_json(path: str, value: dict[str, object]) -> None:
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as json_file:
        json.dump(value, json_file, ensure_ascii=False, indent=2, sort_keys=True)
        json_file.write("\n")
    os.replace(temporary_path, path)


def _validate_extraction_metadata() -> dict[str, object]:
    if not os.path.isfile(EXTRACTION_METADATA_PATH):
        raise RuntimeError(
            "Không tìm thấy metadata Qwen. Hãy chạy extractor.py hoàn tất trước khi index."
        )
    metadata = _read_json(EXTRACTION_METADATA_PATH)
    if metadata.get("status") != "complete":
        raise RuntimeError(
            "Extraction Qwen chưa hoàn tất. Không index artifact có trạng thái in_progress."
        )
    mismatched_keys = [
        key
        for key, expected_value in _expected_metadata().items()
        if metadata.get(key) != expected_value
    ]
    if mismatched_keys:
        raise RuntimeError(
            "Metadata extraction không khớp config hiện tại ở: "
            + ", ".join(mismatched_keys)
        )
    return metadata


def _validate_feature_file(npy_path: str, names_path: str) -> int:
    """Validate before resetting Qdrant, so invalid inputs never destroy a good index."""
    if not os.path.isfile(names_path):
        raise RuntimeError(f"Thiếu filename artifact tương ứng: {names_path}")
    try:
        features = np.load(npy_path, mmap_mode="r")
        filenames = np.load(names_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Không đọc được artifact: {npy_path}") from error

    if features.ndim != 2:
        raise RuntimeError(f"Feature phải là mảng 2 chiều: {npy_path}, nhận {features.shape}")
    if features.shape[1] != VECTOR_SIZE:
        raise RuntimeError(
            f"Sai vector dimension ở {npy_path}: {features.shape[1]}, cần {VECTOR_SIZE}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise RuntimeError(f"Feature phải là float: {npy_path}, nhận {features.dtype}")
    if filenames.ndim != 1 or len(filenames) != features.shape[0]:
        raise RuntimeError(
            f"Số filename không khớp số vector ở {npy_path}: "
            f"{len(filenames)} != {features.shape[0]}"
        )
    if not np.isfinite(features).all():
        raise RuntimeError(f"Feature chứa NaN hoặc Inf: {npy_path}")
    return int(features.shape[0])


# ──────────────────────────────────────────────────────────────────────────────
# Kiểm tra điều kiện tiên quyết
# ──────────────────────────────────────────────────────────────────────────────
print_config()

if not os.path.exists(NPY_DIR):
    print(f"❌ Chưa có thư mục feature Qwen tại: {NPY_DIR}")
    print("👉 Hãy chạy extractor.py trước để sinh ra các file .npy")
    sys.exit(0)

try:
    extraction_metadata = _validate_extraction_metadata()
except RuntimeError as error:
    print(f"❌ {error}")
    sys.exit(1)

npy_files = sorted([
    filename
    for filename in os.listdir(NPY_DIR)
    if filename.endswith(".npy") and filename.startswith("L") and not filename.endswith("_filenames.npy")
])

if not npy_files:
    print(f"⚠️  Không tìm thấy file .npy nào trong: {NPY_DIR}")
    print("👉 Hãy chạy extractor.py trước.")
    sys.exit(0)

# Validate toàn bộ artifact trước khi reset collection cũ.
try:
    validated_counts = {
        npy_filename: _validate_feature_file(
            os.path.join(NPY_DIR, npy_filename),
            os.path.join(NPY_DIR, npy_filename.replace(".npy", "_filenames.npy")),
        )
        for npy_filename in npy_files
    }
except RuntimeError as error:
    print(f"❌ Không thể index Qwen artifact: {error}")
    sys.exit(1)

expected_file_count = extraction_metadata.get("feature_file_count")
expected_vector_count = extraction_metadata.get("vector_count")
if isinstance(expected_file_count, int) and expected_file_count != len(npy_files):
    print(
        "❌ Metadata/file count không khớp. Hãy chạy lại extractor.py để hoàn tất "
        "artifact trước khi index."
    )
    sys.exit(1)
if isinstance(expected_vector_count, int) and expected_vector_count != sum(validated_counts.values()):
    print(
        "❌ Metadata/vector count không khớp. Hãy chạy lại extractor.py để hoàn tất "
        "artifact trước khi index."
    )
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Khởi tạo Qdrant Client (Local Storage — không cần Docker)
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(QDRANT_DB_PATH, exist_ok=True)
print(f"Đang khởi tạo Qdrant DB tại: {QDRANT_DB_PATH}")
client = QdrantClient(path=QDRANT_DB_PATH)

try:
    # Server từ chối index ở trạng thái building, tránh query nhầm trong lúc rebuild.
    _write_json(
        INDEX_METADATA_PATH,
        {
            **_expected_metadata(),
            "collection_name": COLLECTION_NAME,
            "resolved_model_revision": extraction_metadata.get("resolved_model_revision"),
            "status": "building",
        },
    )

    # Reset Collection sau khi mọi .npy đã validate thành công.
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Phát hiện Collection '{COLLECTION_NAME}' đã tồn tại. Đang xóa để nạp lại...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"Đang tạo Collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, Cosine)...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("✅ Tạo Collection thành công!\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Nạp dữ liệu từ các file .npy vào Qdrant (Tối ưu Batch 5000 points)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"Tìm thấy {len(npy_files)} file .npy Qwen. Bắt đầu nạp dữ liệu...\n")

    global_id = 0
    points_buffer = []
    upload_batch_size = 5000

    for index, npy_filename in enumerate(npy_files, 1):
        video_name = npy_filename.removesuffix(".npy")
        npy_path = os.path.join(NPY_DIR, npy_filename)
        names_path = os.path.join(NPY_DIR, f"{video_name}_filenames.npy")

        features = np.load(npy_path)
        image_filenames = [str(name) for name in np.load(names_path, allow_pickle=False).tolist()]

        for frame_offset, vector in enumerate(features):
            frame_name = image_filenames[frame_offset]
            try:
                frame_index = int(os.path.splitext(frame_name)[0])
            except ValueError:
                frame_index = frame_offset

            # Stable IDs make re-indexing idempotent: the same keyframe is updated
            # instead of becoming another searchable point.
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"aic2026-keyframe/{video_name}/{frame_name}",
            ))
            points_buffer.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "video_id": video_name,
                    "frame_id": frame_name,
                    "frame_index": frame_index,
                },
            ))
            global_id += 1

        if len(points_buffer) >= upload_batch_size:
            client.upload_points(
                collection_name=COLLECTION_NAME,
                points=points_buffer,
            )
            print(
                f"  ⚡ [{index}/{len(npy_files)} videos] Đã nạp batch "
                f"{len(points_buffer)} vectors (Tổng: {global_id})"
            )
            points_buffer = []

    if points_buffer:
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points_buffer,
        )
        print(f"  ⚡ Hoàn tất batch cuối {len(points_buffer)} vectors (Tổng: {global_id})")

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    _write_json(
        INDEX_METADATA_PATH,
        {
            **_expected_metadata(),
            "collection_name": COLLECTION_NAME,
            "resolved_model_revision": extraction_metadata.get("resolved_model_revision"),
            "status": "complete",
            "points_count": global_id,
        },
    )

    print(f"\n{'=' * 55}")
    print(f"✅ HOÀN TẤT! DB đang lưu: {collection_info.points_count} vector")
    print(f"   DB path: {QDRANT_DB_PATH}")
finally:
    # Đóng client chủ động để tránh cảnh báo rác của Python khi thoát trên Windows
    client.close()
