from typing import Annotated

from fastapi import APIRouter, Depends

from backend.routers.dependencies import get_vqa_service
from backend.schemas.vqa import VQARequest, VQAResponse
from backend.services.vqa_service import VQAService


router = APIRouter(prefix="/vqa", tags=["vqa"])


@router.post("", response_model=VQAResponse)
async def answer_vqa(
    body: VQARequest,
    service: Annotated[VQAService, Depends(get_vqa_service)],
) -> VQAResponse:
    return await service.answer(body)
