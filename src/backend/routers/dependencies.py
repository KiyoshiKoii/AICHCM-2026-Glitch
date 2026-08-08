from fastapi import Request

from backend.services.search_orchestrator import SearchService
from backend.services.vqa_service import VQAService


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_vqa_service(request: Request) -> VQAService:
    return request.app.state.vqa_service
