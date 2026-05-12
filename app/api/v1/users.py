"""User's endpoints."""

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import CurrentUser
from app.db import queries
from app.models.schemas import UserNameUpdate

router = APIRouter(tags=["users"])


@router.get("/users/me")
async def get_me(user_id: CurrentUser) -> dict:
    """Return the current authenticated user's profile."""
    user = await queries.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"id": user["id"], "name": user["name"], "email": user["email"]}


@router.patch("/users/me")
async def update_me(body: UserNameUpdate, user_id: CurrentUser) -> dict:
    """Update the current user's display name."""
    await queries.update_user_name(user_id, body.name)
    return {"ok": True}


@router.get("/users/me/ws-token")
async def get_ws_token(request: Request, user_id: CurrentUser) -> dict:
    """Return the raw accessToken JWT for use as a WebSocket query parameter.

    Browsers cannot send custom headers on WebSocket connections, and cookies
    are not forwarded when the WS connects to a different port (localhost:8001
    vs the Next.js app at localhost:3000). This endpoint validates the cookie
    and hands back the token so the client can pass it as ?token=<jwt>.
    """
    token = request.cookies.get("accessToken", "")
    return {"token": token}
