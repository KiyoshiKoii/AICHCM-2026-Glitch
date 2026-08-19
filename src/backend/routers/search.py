from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from backend.core.errors import (
    ImageTooLargeError,
    ImageValidationError,
    UnsupportedImageTypeError,
)
from backend.schemas.search import (
    TemporalEventSearchRequest,
    TemporalVideoSearchRequest,
    TextSearchRequest,
    TextSearchResponse,
)
from backend.routers.dependencies import get_search_service
from backend.services.search_orchestrator import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/text", response_model=TextSearchResponse)
async def search_text(
    body: TextSearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> TextSearchResponse:
    return await service.search_text(
        body.query,
        body.top_k,
        use_rerank=body.use_rerank,
        text_weight=body.text_weight,
        visual_weight=body.visual_weight,
        batch_ids=body.batch_ids,
        video_ids=body.video_ids,
    )


@router.post("/image", response_model=TextSearchResponse)
async def search_image(
    request: Request,
    image_file: Annotated[UploadFile, File()],
    service: Annotated[SearchService, Depends(get_search_service)],
    top_k: int = Form(100),
) -> TextSearchResponse:
    content_type = image_file.content_type or "application/octet-stream"
    if not (content_type.startswith("image/jpeg") or content_type.startswith("image/png") or content_type.startswith("image/jpg")):
        raise UnsupportedImageTypeError("Uploaded file must be .jpg, .jpeg, or .png")

    max_bytes = request.app.state.settings.max_image_bytes
    content = await image_file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ImageTooLargeError(f"Image exceeds the {max_bytes}-byte limit")
    if not content:
        raise ImageValidationError("Uploaded image is empty")

    return await service.search_image(
        filename=image_file.filename or "query-image",
        content=content,
        content_type=content_type,
        top_k=top_k
    )


@router.post("/temporal-events")
async def search_temporal_events(
    body: TemporalEventSearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> dict:
    return await service.search_temporal_events(
        body.query,
        batch_ids=body.batch_ids,
        video_ids=body.video_ids,
        top_k_videos=body.top_k_videos,
    )


@router.post("/temporal-videos")
async def search_temporal_videos(
    body: TemporalVideoSearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> dict:
    return await service.search_temporal_videos(
        body.query,
        batch_ids=body.batch_ids,
        video_ids=body.video_ids,
        top_k_videos=body.top_k_videos,
        summary_weight=body.summary_weight,
        kis_weight=body.kis_weight,
    )
