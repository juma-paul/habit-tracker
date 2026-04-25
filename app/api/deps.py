"""FasfAPI dependencies."""

from typing import Annotated

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from fastapi import Depends, HTTPException, Request

from app.core.config import get_settings
from app.db import queries


async def get_current_user_id(request: Request) -> int:
    """
    Verify the httpOnly accessToken cookie set by AuthKit.

    - Decodes the JWT locally using the shared JWT_SECRET
    - Extracts `userId` claim (Authkit's UUID for the user)
    - Auto-provisions a local user row on first login
    - Returns the internal integer user_id for use in queries

    No AuthKit network call needed — verification is pure cryptography
    """
    token = request.cookies.get("accessToken")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    external_id: str = payload["userId"]
    email: str = payload.get("email", "")

    user = await queries.get_or_create_user(external_id, email)

    return user["id"]

# Type alias for cleaner endpoint signatures
CurrentUser = Annotated[int, Depends(get_current_user_id)]