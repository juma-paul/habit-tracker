"""Chat endpoint tests."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.schemas import AgentResponse, AgentStatus


@pytest.fixture
def mock_agent():
    """Mock the agent to avoid real LLM calls in tests."""
    with patch("app.api.v1.chat.run_agent", new_callable=AsyncMock) as mock:
        mock.return_value = AgentResponse(
            status=AgentStatus.success,
            message="Created habit: walking",
            data={"habit": {"id": 1, "name": "walking"}},
        )
        yield mock


def test_chat_authenticated(auth_client, mock_agent):
    """POST /chat with valid cookie returns agent response."""
    response = auth_client.post(
        "/api/v1/chat",
        json={"message": "Create a habit called walking"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_chat_unauthenticated(client):
    """POST /chat without cookie returns 401."""
    response = client.post("/api/v1/chat", json={"message": "Hello"})
    assert response.status_code == 401


def test_chat_empty_message(auth_client):
    """POST /chat with empty message fails validation."""
    response = auth_client.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_routes_to_agent(auth_client, mock_agent):
    """POST /chat always routes through the agent regardless of message content."""
    response = auth_client.post("/api/v1/chat", json={"message": "help me"})
    assert response.status_code == 200
    mock_agent.assert_called_once()
