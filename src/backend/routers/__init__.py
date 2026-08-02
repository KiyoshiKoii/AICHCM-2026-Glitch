from fastapi import APIRouter

from backend.routers.search import router as search_router
from backend.routers.frames import router as frames_router

api_router = APIRouter()
api_router.include_router(search_router)
api_router.include_router(frames_router)
