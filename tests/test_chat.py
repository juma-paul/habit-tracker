"""Chat endpoint tests."""

import pytest
from unittest.mock import patch

from app.models.schemas import AgentResponse, AgentStatus


@pytest.fixture
def mock_agent():
    """Mock the agent to avoid OpenAI calls in tests."""
    with patch("app.api.v1.chat.run_agent") as mock:
        mock.return_value = AgentResponse(
            status=AgentStatus.success,
            message="Created habit: walking",
            data={"habit": {"id": 1, "name": "walking"}}
        )
        yield mock


def test_chat_authenticated(auth_client, mock_agent):
    """POST /chat with valid cookie returns agent response."""
    response = auth_client.post(
        "/api/v1/chat",
        json={"message": "Create a habit called walking"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_chat_unauthenticated(client):
    """POST /chat without cookie returns 401."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello"}
    )
    assert response.status_code == 401


def test_chat_empty_message(auth_client):
    """POST /chat with empty message fails validation."""
    response = auth_client.post(
        "/api/v1/chat",
        json={"message": ""}
    )
    assert response.status_code == 422


def test_help_command(auth_client):
    """POST /chat with /help returns help content without calling agent."""
    with patch("app.api.v1.chat.run_agent") as mock:
        response = auth_client.post(
            "/api/v1/chat",
            json={"message": "/help"}
        )
        assert response.status_code == 200
        mock.assert_not_called()