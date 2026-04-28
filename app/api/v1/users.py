"""User's endpoints."""

from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentUser
from app.db import queries


router = APIRouter(tags=["users"])

@router.get("/users/me")
async def get_me(user_id: CurrentUser) -> dict:
    """Return the current authenticated user's profile."""
    user = await queries.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"id": user["id"], "name": user["name"], "email": user["email"]}