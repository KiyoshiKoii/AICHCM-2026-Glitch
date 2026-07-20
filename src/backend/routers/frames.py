from fastapi import APIRouter, Request

from backend.models.schemas import FrameContextResponse
from backend.utils.frame_id import build_frame_context

router = APIRouter(prefix="/frames", tags=["frames"])


@router.get("/context/{frame_id}", response_model=FrameContextResponse)
async def frame_context(request: Request, frame_id: str) -> FrameContextResponse:
    return build_frame_context(
        frame_id,
        thumbnail_base_url=request.app.state.settings.thumbnail_base_url,
        radius=5,
    )
