"""Tests for streaming chat and WebSocket voice."""

import pytest
from unittest.mock import patch
from starlette.websockets import WebSocketDisconnect


def make_mock_stream(*chunks):
    """Create an async generator that yields the given chunks."""
    async def _gen(*args, **kwargs):
        for chunk in chunks:
            yield chunk
    return _gen


def test_chat_stream_authenticated(auth_client):
    """POST /chat/stream with valid cookie returns SSE stream."""
    with patch("app.api.v1.chat.run_agent_stream", new=make_mock_stream("Hello ", "world")):
        response = auth_client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"}
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_chat_stream_unauthenticated(client):
    """POST /chat/stream without cookie returns 401."""
    response = client.post(
        "/api/v1/chat/stream",
        json={"message": "Hello"}
    )
    assert response.status_code == 401


def test_websocket_no_cookie(client):
    """WebSocket connection without accessToken cookie is rejected."""
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/ws/voice"):
            pass


def test_websocket_invalid_cookie(client):
    """WebSocket connection with a tampered cookie is rejected."""
    client.cookies.set("accessToken", "bad.token.here")
    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/api/v1/ws/voice"):
            pass


def test_websocket_ping_pong(auth_client):
    """WebSocket connection with valid cookie accepts ping and returns pong."""
    with auth_client.websocket_connect("/api/v1/ws/voice") as ws:
        # Server sends conversation_id immediately on connect
        first = ws.receive_json()
        assert first["type"] == "conversation_id"
        ws.send_json({"type": "ping"})
        response = ws.receive_json()
        assert response["type"] == "pong"


def test_websocket_process_without_audio(auth_client):
    """Processing with no audio buffer returns an error message."""
    with auth_client.websocket_connect("/api/v1/ws/voice") as ws:
        ws.send_json({"type": "process"})
        response = ws.receive_json()
        # First message is conversation_id, then the error
        if response["type"] == "conversation_id":
            response = ws.receive_json()
        assert response["type"] == "error"
        assert "No audio" in response["message"]
