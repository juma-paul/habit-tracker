"""Authentication tests — auth is handled by AuthKit, not habit tracker.

These tests verify that the habit tracker correctly:
- Rejects requests with no cookie
- Accepts requests with a valid accessToken cookie
- Auto-provisions users on first authenticated request
"""


def test_me_authenticated(auth_client):
    """GET /users/me with valid cookie returns user profile."""
    response = auth_client.get("/api/v1/users/me")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "email" in data
    assert "name" in data

def test_me_unauthenticated(client):
    """GET /users/me without cookie returns 401."""
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401


def test_me_invalid_token(client):
    """GET /users/me with a tampered token returns 401."""
    client.cookies.set("accessToken", "invalid.token.here")
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401


def test_user_auto_provisioned(auth_client, test_user):
    """First authenticated request creates the user row in the DB."""
    response = auth_client.get("/api/v1/users/me")
    assert response.status_code == 200
    assert response.json()["id"] == test_user["id"]