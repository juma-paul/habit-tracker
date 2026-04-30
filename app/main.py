"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.connection import init_pool, close_pool
from app.api.v1 import router as v1_router


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    setup_logging()
    logger.info("Starting Habit Tracking API")
    await init_pool()
    logger.info("Database pool initialized")
    yield
    logger.info("🛑 Shutting down Habit Tracking API")
    await close_pool()
    logger.info("Database pool closed")


app = FastAPI(
    title="Habit Tracking Agent",
    description="AI-powered habit tracking web app with text and voice.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if get_settings().is_dev else None,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(v1_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
