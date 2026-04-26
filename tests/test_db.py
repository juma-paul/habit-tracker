"""Database query tests."""
import uuid
import pytest

from app.db import queries
from tests.conftest import run


@pytest.fixture
def user_id():
    """Create a test user via get_or_create_user and return their DB id."""
    external_id = str(uuid.uuid4())
    email = f"db-test-{external_id}@example.com"
    user = run(queries.get_or_create_user(external_id, email, "DB Test User"))
    return user["id"]


def test_create_habit(user_id):
    habit = run(queries.create_habit(user_id, "walking", 10000, "steps", "daily"))
    assert habit["name"] == "walking"
    assert habit["target"] == 10000
    assert habit["user_id"] == user_id


def test_get_habits(user_id):
    run(queries.create_habit(user_id, "reading", 30, "minutes"))
    habits = run(queries.get_habits(user_id))
    assert isinstance(habits, list)
    assert len(habits) >= 1


def test_habit_ownership(user_id):
    habit = run(queries.create_habit(user_id, "owned", 100))

    other_id = str(uuid.uuid4())
    other_user = run(queries.get_or_create_user(other_id, f"{other_id}@example.com"))

    result = run(queries.get_habit(habit["id"], other_user["id"]))
    assert result is None


def test_soft_delete(user_id):
    habit = run(queries.create_habit(user_id, "to-delete"))
    run(queries.delete_habit(habit["id"], user_id))

    habits = run(queries.get_habits(user_id))
    assert all(h["id"] != habit["id"] for h in habits)


def test_log_with_ownership(user_id):
    habit = run(queries.create_habit(user_id, "logtest"))
    log = run(queries.create_log(habit["id"], user_id, 42, "test note"))

    assert log["value"] == 42
    assert log["habit_id"] == habit["id"]


def test_log_denied_for_other_user(user_id):
    habit = run(queries.create_habit(user_id, "private"))

    other_id = str(uuid.uuid4())
    other_user = run(queries.get_or_create_user(other_id, f"{other_id}@example.com"))

    with pytest.raises(ValueError, match="access denied"):
        run(queries.create_log(habit["id"], other_user["id"], 100))


def test_get_or_create_user_idempotent():
    external_id = str(uuid.uuid4())
    email = f"idem-{external_id}@example.com"

    user1 = run(queries.get_or_create_user(external_id, email))
    user2 = run(queries.get_or_create_user(external_id, email))

    assert user1["id"] == user2["id"]
