from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Request, UploadFile

from backend.core.errors import (
    ImageTooLargeError,
    ImageValidationError,
    UnsupportedImageTypeError,
)
from backend.models.schemas import TextSearchRequest, TextSearchResponse
from backend.routers.dependencies import get_search_service
from backend.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/text", response_model=TextSearchResponse)
async def search_text(
    body: TextSearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> TextSearchResponse:
    return await service.search_text(body.query, body.top_k)


@router.post("/image")
async def search_image(
    request: Request,
    file: Annotated[UploadFile, File()],
    service: Annotated[SearchService, Depends(get_search_service)],
) -> Any:
    content_type = file.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise UnsupportedImageTypeError("Uploaded file must have an image/* content type")

    max_bytes = request.app.state.settings.max_image_bytes
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ImageTooLargeError(f"Image exceeds the {max_bytes}-byte limit")
    if not content:
        raise ImageValidationError("Uploaded image is empty")

    return await service.search_image(
        filename=file.filename or "query-image",
        content=content,
        content_type=content_type,
    )
