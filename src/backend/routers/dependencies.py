from fastapi import Request

from backend.services.search_orchestrator import SearchService


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service
