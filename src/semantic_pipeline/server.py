# Task 3: API Server (FastAPI) - Port 8002
# Viết endpoint POST /internal/search/text

import os
import socket
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

try:
    # Chạy trực tiếp: python src/semantic_pipeline/server.py
    from database import TextDatabase
except ImportError:
    # Import như package (pytest ở Task 5, hoặc uvicorn src.semantic_pipeline.server:app)
    from .database import TextDatabase


FilterValue = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SearchFilters(BaseModel):
    """Optional exact metadata filters, available with Elasticsearch backend."""

    model_config = ConfigDict(extra="forbid")

    time_of_day: Literal["unknown", "morning", "afternoon", "evening", "night"] | None = None
    setting: Literal["unknown", "indoor", "outdoor"] | None = None
    locations: list[FilterValue] | None = Field(default=None, min_length=1)
    objects: list[FilterValue] | None = Field(default=None, min_length=1)
    actions: list[FilterValue] | None = Field(default=None, min_length=1)
    colors: list[FilterValue] | None = Field(default=None, min_length=1)
    code_language: Literal["unknown", "sql"] | None = None
    code_patterns: list[FilterValue] | None = Field(default=None, min_length=1)
    spatial_relations: list["SpatialRelationFilter"] | None = Field(
        default=None, min_length=1
    )


class SpatialRelationFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: FilterValue
    predicate: Literal[
        "left_of", "right_of", "above", "below", "overlapping"
    ]
    object: FilterValue


class SearchRequest(BaseModel):
    """Payload Dev 3 gửi sang (theo API Contract v1, mục 5).

    {"keywords": ["pink teddy bear", "keychain", "dropping"], "top_k": 200}
    """

    # min_length=1: mảng rỗng gần như chắc chắn là bug bên Dev 3 (LLM không sinh
    # được keyword nào). Trả 422 để họ thấy ngay, thay vì 200 + data rỗng khiến
    # họ tưởng "không tìm thấy gì" và đi soi nhầm chỗ.
    keywords: list[str] = Field(min_length=1)

    # Mặc định 200 theo contract. Chặn trên 1000 để Dev 3 không vô tình gửi
    # top_k=999999 làm server phải sort/serialize cả corpus.
    top_k: int = Field(default=200, ge=1, le=1000)
    filters: SearchFilters | None = None


class FrameResult(BaseModel):
    """Một frame trong kết quả — đúng 4 field của API Contract."""

    frame_id: str
    score: float
    video_name: str
    frame_index: int


class SearchResponse(BaseModel):
    """Response bọc ngoài, đúng API Contract v1 mục 5."""

    status: str = "success"
    data: list[FrameResult]


# Giữ index BM25 trên RAM, dùng lại cho mọi request.
db = None


def _create_search_backend():
    """Select BM25 by default; Elasticsearch is an opt-in Task 4 upgrade."""
    backend_name = os.getenv("SEMANTIC_SEARCH_BACKEND", "bm25").lower()
    if backend_name == "bm25":
        return TextDatabase()
    if backend_name == "elasticsearch":
        try:
            from elasticsearch_backend import ElasticsearchBackend
        except ImportError:
            from .elasticsearch_backend import ElasticsearchBackend
        backend = ElasticsearchBackend()
        if not backend.ping():
            raise RuntimeError("Elasticsearch backend is configured but not reachable")
        return backend
    raise ValueError(
        "SEMANTIC_SEARCH_BACKEND must be either 'bm25' or 'elasticsearch'"
    )


def _document_count() -> int:
    if db is None:
        return 0
    if hasattr(db, "document_count"):
        return db.document_count()
    return len(db.records)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Dựng BM25 index MỘT LẦN lúc server khởi động.

    Nếu khởi tạo TextDatabase bên trong hàm xử lý request thì mỗi lần Dev 3 gọi
    API, server sẽ đọc lại metadata.json và dựng lại toàn bộ index -> rất chậm.
    """
    global db
    db = _create_search_backend()
    print(
        f"[startup] {type(db).__name__} sẵn sàng: {_document_count()} documents"
    )

    yield  # server phục vụ request ở đây

    db = None
    print("[shutdown] Đã giải phóng index")


app = FastAPI(
    title="Semantic Pipeline API (Dev 2)",
    description="Semantic search trên caption, OCR, entity/spatial và code metadata.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Dev 3 kiểm tra service sống chưa, index đã nạp bao nhiêu document."""
    if db is None:
        raise HTTPException(status_code=503, detail="BM25 index chưa sẵn sàng")
    return {
        "status": "ok",
        "backend": type(db).__name__,
        "documents": _document_count(),
    }


# Dùng `def` chứ KHÔNG phải `async def`: FastAPI chạy hàm def thường trong
# threadpool riêng, nên BM25 (nặng CPU) không chặn event loop. Đổi sang
# `async def` sẽ block event loop và mọi request khác phải xếp hàng chờ.
@app.post("/internal/search/text", response_model=SearchResponse)
def search_text(request: SearchRequest) -> SearchResponse:
    """Dev 3 gửi mảng keywords tiếng Anh (đã LLM mở rộng) -> trả frame khớp nhất.

    Điểm BM25 để NGUYÊN, không chuẩn hóa (đúng luật task.md) — Dev 3 tự dùng
    thứ hạng (RRF) để gộp với điểm của Dev 1.
    """
    if db is None:  # lifespan chưa chạy xong
        raise HTTPException(status_code=503, detail="BM25 index chưa sẵn sàng")

    filters = (
        request.filters.model_dump(exclude_none=True)
        if request.filters is not None
        else {}
    )
    if filters and not getattr(db, "supports_filters", False):
        raise HTTPException(
            status_code=400,
            detail=(
                "Metadata filters require SEMANTIC_SEARCH_BACKEND=elasticsearch"
            ),
        )

    if filters:
        results = db.search(request.keywords, request.top_k, filters=filters)
    else:
        results = db.search(request.keywords, request.top_k)
    return SearchResponse(status="success", data=results)


PORT = 8002  # cố định theo task.md (Dev 1 = 8001, Dev 3 = 8000)


def _dual_stack_socket(port: int) -> socket.socket:
    """Socket nghe được CẢ IPv6 (::1) lẫn IPv4 (127.0.0.1).

    Vì sao cần: Windows phân giải "localhost" thành ::1 (IPv6) TRƯỚC. API Contract
    quy định Dev 3 gọi http://localhost:8002, nên:
      - Bind "0.0.0.0" (chỉ IPv4) -> client thử IPv6 trước, chờ timeout ~2 GIÂY
        rồi mới fallback IPv4. Dev 3 sẽ tưởng BM25 chậm (thực tế chỉ tốn 0.06ms).
      - Bind "::" bằng uvicorn.run() -> trên Windows ra socket IPv6-ONLY,
        127.0.0.1 bị "connection refused".
    Chỉ có cách tự tạo socket rồi tắt cờ IPV6_V6ONLY mới nghe được cả hai.
    """
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)  # <- mấu chốt
    sock.bind(("::", port))
    sock.listen()
    return sock


if __name__ == "__main__":
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, log_level="info"))
    print(f"[startup] Nghe tren localhost:{PORT} va 127.0.0.1:{PORT} (dual-stack)")
    server.run(sockets=[_dual_stack_socket(PORT)])
