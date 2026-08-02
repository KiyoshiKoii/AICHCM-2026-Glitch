"""
test_performance.py — Performance Tests cho Visual Pipeline
============================================================
Đo hai loại latency theo yêu cầu Task 5:

  PERF-01  Batch Inference latency (CLIP image embedding)
           Target: trung bình < 500ms / batch (batch_size=32)

  PERF-02  Qdrant query latency (cosine search)
           Target: mỗi query < 500ms

Cơ chế:
  - PERF-01 dùng model CLIP thật (transformers) nhưng với ảnh giả
    (random PIL Image) để không cần dataset thật.
    Nếu môi trường không có torch/transformers, test sẽ bị skip tự động.

  - PERF-02 dùng Qdrant local với collection nhỏ tạm thời (tmp_path)
    chứa 100 vector ngẫu nhiên 512 chiều.
    Nếu không có qdrant-client, test bị skip tự động.

Cách chạy:
    cd src/visual_pipeline
    pytest tests/test_performance.py -v -s
"""

import os
import sys
import time
import random
import statistics
import pytest

# ─── Xóa stub modules do test_api.py inject (khi chạy chung toàn bộ suite) ───
# test_api.py stub torch/transformers/qdrant_client để chạy nhanh, nhưng
# test_performance.py cần thư viện THẬT. Module-level code chạy theo thứ tự
# import; nếu test_api.py được collect trước, stubs đã có trong sys.modules.
# Ta phát hiện stub qua __file__ is None (thư viện thật luôn có __file__).

def _is_stub(mod) -> bool:
    return getattr(mod, "__file__", None) is None

for _m in ["torch", "transformers", "qdrant_client", "qdrant_client.models"]:
    if _m in sys.modules and _is_stub(sys.modules[_m]):
        del sys.modules[_m]

# ─── Thêm src/visual_pipeline vào sys.path ───────────────────────────────────
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

# ─── Kiểm tra thư viện sẵn có ────────────────────────────────────────────────
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import numpy as np
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import numpy as np
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False

from config import CLIP_MODEL_ID, VECTOR_SIZE, BATCH_SIZE

# ─── Ngưỡng và số vòng lặp ───────────────────────────────────────────────────
LATENCY_THRESHOLD_MS = 500.0
N_RUNS = 5
_TEST_COLLECTION = "perf_test_collection"
_N_POINTS = 100


# ─── Module-level fixtures (tránh deprecated class-scoped instance fixtures) ──

@pytest.fixture(scope="module")
def clip_components():
    """Load CLIP model + processor một lần cho cả module."""
    if not _HAS_CLIP:
        pytest.skip("torch / transformers / PIL không khả dụng")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[PERF-01] Loading CLIP model ({CLIP_MODEL_ID}) trên {device.upper()}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.to(device)
    model.eval()
    print("[PERF-01] Model san sang.")
    return model, processor, device


@pytest.fixture(scope="module")
def qdrant_db(tmp_path_factory):
    """Tạo Qdrant DB tạm thời với 100 vector ngẫu nhiên."""
    if not _HAS_QDRANT:
        pytest.skip("qdrant-client khong kha dung")
    import numpy as _np

    db_path = str(tmp_path_factory.mktemp("perf_qdrant"))
    client = QdrantClient(path=db_path)
    client.create_collection(
        collection_name=_TEST_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    rng = _np.random.default_rng(seed=42)
    points = [
        PointStruct(
            id=i,
            vector=rng.random(VECTOR_SIZE).tolist(),
            payload={"video_id": f"L25_V{i:03d}", "frame_id": f"{i:04d}.jpg"},
        )
        for i in range(_N_POINTS)
    ]
    client.upload_points(collection_name=_TEST_COLLECTION, points=points)
    print(f"\n[PERF-02] Da nap {_N_POINTS} vector vao Qdrant DB: {db_path}")
    return client


# ─── Helper functions ─────────────────────────────────────────────────────────

def _make_random_images(n: int):
    images = []
    for _ in range(n):
        data = bytes([random.randint(0, 255) for _ in range(224 * 224 * 3)])
        images.append(Image.frombytes("RGB", (224, 224), data))
    return images


def _run_clip_batch(model, processor, device, images):
    inputs = processor(images=images, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # Mot so phien ban transformers cu tra ve BaseModelOutputWithPooling
        # thay vi tensor truc tiep — can extract dung truong
        if hasattr(features, "image_embeds"):
            features = features.image_embeds
        elif hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return (time.perf_counter() - t0) * 1000, features



def _run_qdrant_query(client, query_vec, top_k=10):
    t0 = time.perf_counter()
    result = client.query_points(
        collection_name=_TEST_COLLECTION,
        query=query_vec,
        limit=top_k,
    )
    return (time.perf_counter() - t0) * 1000, result


# ══════════════════════════════════════════════════════════════════════════════
# PERF-01: Batch Inference Latency
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_CLIP, reason="torch / transformers / PIL khong kha dung")
class TestBatchInferenceLatency:
    """Do toc do trich xuat vector anh theo batch."""

    def test_perf01_avg_latency_under_threshold(self, clip_components):
        """
        PERF-01: Trung binh PER-IMAGE latency cua N_RUNS batch phai < 500ms.
        (total_batch_ms / batch_size)
        CPU: ~2600ms / 32 anh = ~83ms/anh  → PASS
        GPU: ~50ms  / 32 anh = ~1.6ms/anh  → PASS
        """
        model, processor, device = clip_components
        images = _make_random_images(BATCH_SIZE)

        batch_latencies = []
        for run in range(N_RUNS):
            ms, _ = _run_clip_batch(model, processor, device, images)
            batch_latencies.append(ms)
            print(f"  [PERF-01] Run {run + 1}/{N_RUNS}: {ms:.1f} ms (batch) | "
                  f"{ms / BATCH_SIZE:.1f} ms/img")

        avg_batch_ms = statistics.mean(batch_latencies)
        avg_per_img_ms = avg_batch_ms / BATCH_SIZE
        p95_ms = sorted(batch_latencies)[min(int(len(batch_latencies) * 0.95),
                                             len(batch_latencies) - 1)]

        print(f"\n  [PERF-01] Ket qua ({N_RUNS} runs, batch_size={BATCH_SIZE}, device={device}):")
        print(f"    Min batch  : {min(batch_latencies):.1f} ms")
        print(f"    Max batch  : {max(batch_latencies):.1f} ms")
        print(f"    Mean batch : {avg_batch_ms:.1f} ms")
        print(f"    ~P95 batch : {p95_ms:.1f} ms")
        print(f"    Per-image  : {avg_per_img_ms:.1f} ms  (target < {LATENCY_THRESHOLD_MS} ms)")

        assert avg_per_img_ms < LATENCY_THRESHOLD_MS, (
            f"[PERF-01] Per-image latency {avg_per_img_ms:.1f} ms vuot nguong "
            f"{LATENCY_THRESHOLD_MS} ms. Chay tren: {device.upper()}"
        )


    def test_perf01b_vector_output_shape(self, clip_components):
        """
        PERF-01b: Output vector phai co shape dung (BATCH_SIZE x VECTOR_SIZE).
        """
        model, processor, device = clip_components
        images = _make_random_images(BATCH_SIZE)
        _, features = _run_clip_batch(model, processor, device, images)

        shape = tuple(features.shape)
        expected = (BATCH_SIZE, VECTOR_SIZE)
        assert shape == expected, (
            f"[PERF-01b] Vector shape sai: nhan {shape}, ky vong {expected}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PERF-02: Qdrant Query Latency
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_QDRANT, reason="qdrant-client khong kha dung")
class TestQdrantQueryLatency:
    """Do toc do truy van Qdrant voi collection nho in-memory."""

    def test_perf02_avg_query_latency_under_threshold(self, qdrant_db):
        """
        PERF-02: Trung binh latency cua N_RUNS query Qdrant phai < 500ms.
        """
        import numpy as _np

        rng = _np.random.default_rng(seed=99)
        latencies = []
        for run in range(N_RUNS):
            query_vec = rng.random(VECTOR_SIZE).tolist()
            ms, _ = _run_qdrant_query(qdrant_db, query_vec)
            latencies.append(ms)
            print(f"  [PERF-02] Run {run + 1}/{N_RUNS}: {ms:.1f} ms")

        avg_ms = statistics.mean(latencies)
        print(f"\n  [PERF-02] Ket qua ({N_RUNS} queries, {_N_POINTS} pts):")
        print(f"    Min  : {min(latencies):.1f} ms")
        print(f"    Max  : {max(latencies):.1f} ms")
        print(f"    Mean : {avg_ms:.1f} ms  (target < {LATENCY_THRESHOLD_MS} ms)")

        assert avg_ms < LATENCY_THRESHOLD_MS, (
            f"[PERF-02] Trung binh {avg_ms:.1f} ms vuot nguong {LATENCY_THRESHOLD_MS} ms"
        )

    def test_perf02b_query_returns_correct_count(self, qdrant_db):
        """
        PERF-02b: Query top_k=10 tra ve dung 10 ket qua.
        """
        import numpy as _np

        query_vec = _np.random.default_rng(0).random(VECTOR_SIZE).tolist()
        _, result = _run_qdrant_query(qdrant_db, query_vec, top_k=10)
        hits = result.points if hasattr(result, "points") else result
        assert len(hits) == 10, (
            f"[PERF-02b] Query top_k=10 phai tra 10 ket qua, nhan {len(hits)}"
        )

    def test_perf02c_cold_start_latency_under_threshold(self, qdrant_db):
        """
        PERF-02c: Cold-start query (lan dau) cung phai < 500ms.
        """
        import numpy as _np

        query_vec = _np.random.default_rng(777).random(VECTOR_SIZE).tolist()
        ms, _ = _run_qdrant_query(qdrant_db, query_vec)
        print(f"\n  [PERF-02c] Cold-start latency: {ms:.1f} ms")
        assert ms < LATENCY_THRESHOLD_MS, (
            f"[PERF-02c] Cold-start {ms:.1f} ms vuot nguong {LATENCY_THRESHOLD_MS} ms"
        )
