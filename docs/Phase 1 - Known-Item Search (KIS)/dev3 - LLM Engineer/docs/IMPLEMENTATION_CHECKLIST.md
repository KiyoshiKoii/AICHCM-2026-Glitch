# Dev 3 implementation checklist

## Completed work

- [x] **LLM parser** — `src/backend/services/llm_parser.py`
  - Vietnamese query to English `visual_prompt` and synonym-rich
    `semantic_keywords`.
  - Ollama structured output with Pydantic JSON Schema and temperature `0`.
  - Defensive extraction for fenced or prefixed JSON.
- [x] **POST `/search/text`** — `src/backend/routers/search.py` and
  `src/backend/services/search_service.py`
  - Dev 1 and Dev 2 calls execute concurrently with `asyncio.gather`.
  - Results are normalized, deduplicated by `frame_id`, fused by RRF, limited
    to 20, and assigned thumbnail URLs.
- [x] **POST `/search/image`** — `src/backend/routers/search.py`
  - Validates image MIME/size and forwards the original bytes only to Dev 1.
  - Does not invoke the parser or Dev 2.
- [x] **GET `/frames/context/{frame_id}`** —
  `src/backend/routers/frames.py` and `src/backend/utils/frame_id.py`
  - Parses the numeric suffix and returns offsets `-5..+5` while preserving
    zero-padding.
- [x] **Unit and API tests** — `tests/`
  - Covers RRF, malformed LLM JSON, the Ollama request schema, concurrent
    fan-out, upstream normalization, image-only routing, upload validation,
    temporal navigation, and public endpoints.
- [x] **Modular FastAPI structure** — all backend source is under
  `src/backend/` with separate `routers/`, `services/`, and `utils/` packages.

## Integration assumptions to confirm with Dev 1 and Dev 2

1. Both text APIs accept a JSON `query` plus `top_k`.
2. Dev 2 additionally accepts `keywords`.
3. Results expose `frame_id` or `id`; optional scores use `score` or
   `similarity`.
4. Dev 1 image search accepts multipart field `file` and returns JSON.
5. Thumbnail files resolve as `{THUMBNAIL_BASE_URL}/{frame_id}.jpg`.

Only `src/backend/services/pipeline_clients.py` and environment variables need
adjustment if the final internal contracts differ.

