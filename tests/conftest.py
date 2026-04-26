"""Pytest configuration and fixtures."""
import asyncio
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
from app.db.connection import init_pool, close_pool


# Module-level reference so run() uses the same loop the pool was opened on
_session_loop: asyncio.AbstractEventLoop | None = None


# Single event loop shared across the entire test session
@pytest.fixture(scope="session")
def event_loop():
    global _session_loop
    loop = asyncio.new_event_loop()
    _session_loop = loop
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_db(event_loop):
    """Initialize async database pool once for all tests."""
    event_loop.run_until_complete(init_pool())
    yield
    event_loop.run_until_complete(close_pool())


def run(coro):
    """Run an async coroutine on the session event loop."""
    assert _session_loop is not None, "session event loop not initialized"
    return _session_loop.run_until_complete(coro)


@pytest.fixture
def client():
    """Unauthenticated test client."""
    return TestClient(app)


@pytest.fixture
def test_user() -> dict:
    """Create a test user via get_or_create_user and return their data."""
    from app.db import queries

    external_id = str(uuid.uuid4())
    email = f"test-{external_id}@example.com"
    user = run(queries.get_or_create_user(external_id, email, "Test User"))
    return {"id": user["id"], "external_id": external_id, "email": email}


@pytest.fixture
def auth_cookies(test_user: dict) -> dict:
    """Signed accessToken cookie matching AuthKit's JWT format."""
    settings = get_settings()
    token = jwt.encode(
        {"userId": test_user["external_id"], "email": test_user["email"]},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    return {"accessToken": token}


@pytest.fixture
def auth_client(client: TestClient, auth_cookies: dict) -> TestClient:
    """Test client with accessToken cookie pre-set."""
    client.cookies.update(auth_cookies)
    return client
