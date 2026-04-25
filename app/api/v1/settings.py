"""User settings endpoints."""

from fastapi import APIRouter

from app.db import queries
from app.models.schemas import UserSettingsResponse, SettingsUpdate
from app.api.deps import CurrentUser


router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("", response_model=UserSettingsResponse)
async def get_settings(user_id: CurrentUser) -> UserSettingsResponse:
    """Get current user's settings."""
    settings = await queries.get_user_settings(user_id)
    return UserSettingsResponse(**settings)


@router.patch("", response_model=UserSettingsResponse)
async def update_settings(req: SettingsUpdate, user_id: CurrentUser) -> UserSettingsResponse:
    """Update user settings."""
    updates = req.model_dump(exclude_none=True)
    settings = await queries.update_user_settings(user_id, **updates)
    return UserSettingsResponse(**settings)