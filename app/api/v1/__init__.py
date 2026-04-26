""""API v1 router."""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.voice import router as voice_router
from app.api.v1.websocket import router as ws_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.settings import router as settings_router

router = APIRouter(prefix="/api/v1")

router.include_router(chat_router)
router.include_router(voice_router)
router.include_router(ws_router)
router.include_router(conversations_router)
router.include_router(settings_router)