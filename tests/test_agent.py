"""Tests for the graph agent — covers all critical routing paths."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.graph_agent import run_graph_agent
from app.models.schemas import AgentStatus


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_classifier_result(
    intent: str,
    habit_name: str | None = None,
    habit_target: float | None = None,
    habit_unit: str | None = None,
    habit_frequency: str | None = None,
    habit_value: float | None = None,
    days: int = 7,
):
    """Build a mock pydantic_ai RunResult for the intent classifier."""
    from app.agent.graph_agent import IntentResult

    output = IntentResult(
        intent=intent,
        habit_name=habit_name,
        habit_target=habit_target,
        habit_unit=habit_unit,
        habit_frequency=habit_frequency,
        habit_value=habit_value,
        days=days,
    )
    result = MagicMock()
    result.output = output
    return result


def _mock_formatter():
    """Return a patched _get_formatter that returns 'Done!' without calling the LLM."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=MagicMock(output="Done!"))
    return MagicMock(return_value=mock_agent)


# ─── Existing tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_habit_guard(test_user):
    """Creating a habit that already exists returns a warning — no new habit created.

    Scenario: user has 'water drinking'. They ask to create 'water drinking' again.
    The agent should detect the duplicate and respond with an informative message
    instead of routing to AskCreateConfirmation.
    """
    user_id = test_user["id"]
    existing_habits = [{"name": "water drinking", "target": 8.0, "unit": "glasses"}]

    with (
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(
                    return_value=_make_classifier_result(
                        intent="create", habit_name="water drinking"
                    )
                )
            ),
        ),
        patch(
            "app.agent.graph_agent.tools.list_habits",
            new_callable=AsyncMock,
            return_value={"habits": existing_habits},
        ),
    ):
        response = await run_graph_agent("create water drinking habit", user_id)

    assert response.status == AgentStatus.success
    assert "already have" in response.message.lower()
    assert response.data is None or response.data.get("awaiting") != "create_confirm"


@pytest.mark.asyncio
async def test_target_captured_from_confirmation(test_user):
    """When user confirms creation AND includes target details, they are used.

    Scenario:
    1. Agent previously asked "Would you like me to create 'water drinking'?"
       (state: awaiting=create_confirm, habit_name='water drinking')
    2. User replies "yes with 12 glasses daily"
    The agent should create the habit with target=12, unit='glasses', frequency='daily'.
    """
    user_id = test_user["id"]

    fake_habit = {
        "id": 99,
        "user_id": user_id,
        "name": "water drinking",
        "target": 12.0,
        "unit": "glasses",
        "frequency": "daily",
        "is_deleted": False,
    }
    confirmation_classifier_result = _make_classifier_result(
        intent="create",
        habit_name="water drinking",
        habit_target=12.0,
        habit_unit="glasses",
        habit_frequency="daily",
    )
    mock_create = AsyncMock(return_value=fake_habit)

    with (
        patch("app.agent.graph_agent._confirmed", new_callable=AsyncMock, return_value=True),
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(run=AsyncMock(return_value=confirmation_classifier_result)),
        ),
        patch("app.agent.graph_agent.queries.create_habit", mock_create),
        patch("app.agent.graph_agent.queries.add_message", new_callable=AsyncMock),
    ):
        response = await run_graph_agent(
            message="yes with 12 glasses daily",
            user_id=user_id,
            awaiting="create_confirm",
            context={"habit_name": "water drinking"},
        )
        assert response.status == AgentStatus.success
        mock_create.assert_called_once_with(user_id, "water drinking", 12.0, "glasses", "daily")


# ─── Log a habit (primary use case) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_known_habit(test_user):
    """Logging a known habit calls log_activity with the correct value.

    Scenario: user says "I ran 5km". The habit 'running' already exists.
    No log today yet. Should call tools.log_activity(user_id, "running", 5.0, None).
    """
    user_id = test_user["id"]
    mock_log = AsyncMock(return_value={"log": {"id": 1, "value": 5.0}, "message": "Logged"})

    with (
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(
                    return_value=_make_classifier_result(
                        intent="log", habit_name="running", habit_value=5.0
                    )
                )
            ),
        ),
        patch(
            "app.agent.graph_agent.tools.list_habits",
            new_callable=AsyncMock,
            return_value={"habits": [{"id": 1, "name": "running", "target": 5, "unit": "km", "frequency": "daily"}]},
        ),
        patch(
            "app.agent.graph_agent.queries.find_habit_by_name",
            new_callable=AsyncMock,
            return_value={"id": 1, "name": "running", "user_id": user_id},
        ),
        patch(
            "app.agent.graph_agent.queries.get_today_logs",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("app.agent.graph_agent.tools.log_activity", mock_log),
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent("I ran 5km", user_id)

    assert response.status == AgentStatus.success
    mock_log.assert_called_once_with(user_id, "running", 5.0, None)


@pytest.mark.asyncio
async def test_log_without_value_prompts_for_amount(test_user):
    """Logging a habit without a number asks the user how much they did."""
    user_id = test_user["id"]

    with (
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(
                    return_value=_make_classifier_result(
                        intent="log", habit_name="running", habit_value=None
                    )
                )
            ),
        ),
    ):
        response = await run_graph_agent("I went for a run", user_id)

    assert response.status == AgentStatus.success
    assert response.data is not None
    assert response.data["awaiting"] == "log_value"
    assert response.data["habit_name"] == "running"


# ─── Show progress / list habits ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_habits(test_user):
    """Asking for habits list calls tools.list_habits and returns success."""
    user_id = test_user["id"]
    habits = [
        {"id": 1, "name": "running", "target": 5, "unit": "km", "frequency": "daily"},
        {"id": 2, "name": "water drinking", "target": 8, "unit": "glasses", "frequency": "daily"},
    ]

    with (
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(return_value=_make_classifier_result(intent="list"))
            ),
        ),
        patch(
            "app.agent.graph_agent.tools.list_habits",
            new_callable=AsyncMock,
            return_value={"habits": habits},
        ),
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent("show me my habits", user_id)

    assert response.status == AgentStatus.success


@pytest.mark.asyncio
async def test_get_progress_for_specific_habit(test_user):
    """Progress request for a named habit calls tools.get_progress with that name."""
    user_id = test_user["id"]
    mock_progress = AsyncMock(
        return_value={"habit": "running", "logs": [], "days": 7, "total": 0}
    )

    with (
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(
                    return_value=_make_classifier_result(
                        intent="progress", habit_name="running", days=7
                    )
                )
            ),
        ),
        patch("app.agent.graph_agent.tools.get_progress", mock_progress),
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent("how's my running habit this week?", user_id)

    assert response.status == AgentStatus.success
    mock_progress.assert_called_once_with(user_id, "running", 7)


# ─── Delete habit flow ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_asks_confirmation(test_user):
    """Deleting a habit first asks for confirmation before doing anything."""
    user_id = test_user["id"]

    with (
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(
                    return_value=_make_classifier_result(intent="delete", habit_name="running")
                )
            ),
        ),
        patch(
            "app.agent.graph_agent.tools.list_habits",
            new_callable=AsyncMock,
            return_value={"habits": [{"id": 1, "name": "running", "target": None, "unit": None, "frequency": "daily"}]},
        ),
    ):
        response = await run_graph_agent("delete my running habit", user_id)

    assert response.status == AgentStatus.success
    assert response.data is not None
    assert response.data["awaiting"] == "delete_confirm"
    assert "running" in response.message
    assert "cannot be undone" in response.message.lower() or "permanently" in response.message.lower()


@pytest.mark.asyncio
async def test_delete_confirmed_calls_delete(test_user):
    """When awaiting delete_confirm and user says yes, the habit is deleted."""
    user_id = test_user["id"]
    mock_delete = AsyncMock(
        return_value={"habit": {"id": 1, "name": "running"}, "message": "Deleted"}
    )

    with (
        patch("app.agent.graph_agent._confirmed", new_callable=AsyncMock, return_value=True),
        patch("app.agent.graph_agent.tools.delete_habit", mock_delete),
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent(
            message="yes",
            user_id=user_id,
            awaiting="delete_confirm",
            context={"habit_name": "running"},
        )

    assert response.status == AgentStatus.success
    mock_delete.assert_called_once_with(user_id, "running")


@pytest.mark.asyncio
async def test_delete_cancelled_keeps_habit(test_user):
    """When awaiting delete_confirm and user says no, nothing is deleted."""
    user_id = test_user["id"]
    mock_delete = AsyncMock()

    with (
        patch("app.agent.graph_agent._confirmed", new_callable=AsyncMock, return_value=False),
        patch("app.agent.graph_agent.tools.delete_habit", mock_delete),
    ):
        response = await run_graph_agent(
            message="no",
            user_id=user_id,
            awaiting="delete_confirm",
            context={"habit_name": "running"},
        )

    assert response.status == AgentStatus.success
    assert "running" in response.message
    assert "safe" in response.message.lower()
    mock_delete.assert_not_called()


# ─── delete_disambiguate ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_disambiguate_shown_when_duplicate_names(test_user):
    """When two habits share the same name, a numbered list is returned before deletion.

    This was the bug: before the fix, a dead-end response was returned with no
    awaiting state, so the next message was re-classified from scratch.
    """
    user_id = test_user["id"]
    duplicate_habits = [
        {"id": 10, "name": "water drinking", "target": 8.0, "unit": "glasses", "frequency": "daily"},
        {"id": 20, "name": "water drinking", "target": 12.0, "unit": "glasses", "frequency": "daily"},
    ]

    with (
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(
                    return_value=_make_classifier_result(
                        intent="delete", habit_name="water drinking"
                    )
                )
            ),
        ),
        patch(
            "app.agent.graph_agent.tools.list_habits",
            new_callable=AsyncMock,
            return_value={"habits": duplicate_habits},
        ),
    ):
        response = await run_graph_agent("delete water drinking", user_id)

    assert response.status == AgentStatus.success
    assert response.data is not None
    assert response.data["awaiting"] == "delete_disambiguate"
    assert response.data["habit_ids"] == [10, 20]
    assert "1." in response.message
    assert "2." in response.message


@pytest.mark.asyncio
async def test_delete_disambiguate_select_by_number(test_user):
    """Replying '1' to a disambiguation prompt selects the first habit and asks to confirm."""
    user_id = test_user["id"]
    chosen_habit = {
        "id": 10,
        "name": "water drinking",
        "target": 8.0,
        "unit": "glasses",
        "frequency": "daily",
    }

    with (
        patch(
            "app.agent.graph_agent.queries.get_habit",
            new_callable=AsyncMock,
            return_value=chosen_habit,
        ),
    ):
        response = await run_graph_agent(
            message="1",
            user_id=user_id,
            awaiting="delete_disambiguate",
            context={"habit_ids": [10, 20], "habit_name": "water drinking"},
        )

    assert response.status == AgentStatus.success
    assert response.data is not None
    assert response.data["awaiting"] == "delete_confirm"
    assert response.data["habit_id"] == 10


@pytest.mark.asyncio
async def test_delete_disambiguate_select_by_ordinal(test_user):
    """Replying 'second' to a disambiguation prompt selects the second habit."""
    user_id = test_user["id"]
    chosen_habit = {
        "id": 20,
        "name": "water drinking",
        "target": 12.0,
        "unit": "glasses",
        "frequency": "daily",
    }

    with (
        patch(
            "app.agent.graph_agent.queries.get_habit",
            new_callable=AsyncMock,
            return_value=chosen_habit,
        ),
    ):
        response = await run_graph_agent(
            message="the second one",
            user_id=user_id,
            awaiting="delete_disambiguate",
            context={"habit_ids": [10, 20], "habit_name": "water drinking"},
        )

    assert response.status == AgentStatus.success
    assert response.data["awaiting"] == "delete_confirm"
    assert response.data["habit_id"] == 20


@pytest.mark.asyncio
async def test_delete_disambiguate_invalid_input_reprompts(test_user):
    """Replying with something that's not a number re-prompts instead of crashing."""
    user_id = test_user["id"]

    response = await run_graph_agent(
        message="I don't know",
        user_id=user_id,
        awaiting="delete_disambiguate",
        context={"habit_ids": [10, 20], "habit_name": "water drinking"},
    )

    assert response.status == AgentStatus.success
    assert response.data is not None
    assert response.data["awaiting"] == "delete_disambiguate"


# ─── Multi-turn awaiting states ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_value_awaiting_parses_number(test_user):
    """Replying with a number to a log_value prompt triggers the log flow."""
    user_id = test_user["id"]
    mock_log = AsyncMock(return_value={"log": {"id": 1, "value": 30.0}, "message": "Logged"})

    with (
        patch(
            "app.agent.graph_agent.tools.list_habits",
            new_callable=AsyncMock,
            return_value={"habits": [{"id": 1, "name": "running", "target": 60, "unit": "minutes", "frequency": "daily"}]},
        ),
        patch(
            "app.agent.graph_agent.queries.find_habit_by_name",
            new_callable=AsyncMock,
            return_value={"id": 1, "name": "running", "user_id": user_id},
        ),
        patch("app.agent.graph_agent.queries.get_today_logs", new_callable=AsyncMock, return_value=[]),
        patch("app.agent.graph_agent.tools.log_activity", mock_log),
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent(
            message="30",
            user_id=user_id,
            awaiting="log_value",
            context={"habit_name": "running"},
        )

    assert response.status == AgentStatus.success
    mock_log.assert_called_once_with(user_id, "running", 30.0, None)


@pytest.mark.asyncio
async def test_log_value_awaiting_rejects_non_number(test_user):
    """Replying without a number to a log_value prompt re-prompts with clarification."""
    user_id = test_user["id"]

    response = await run_graph_agent(
        message="yesterday",
        user_id=user_id,
        awaiting="log_value",
        context={"habit_name": "running"},
    )

    assert response.status == AgentStatus.clarification
    assert response.data is not None
    assert response.data["awaiting"] == "log_value"


@pytest.mark.asyncio
async def test_create_confirm_interrupted_by_list(test_user):
    """While waiting for a create confirmation, asking for habits routes to list — not create.

    This documents the exact user conversation that prompted concern:
      1. Agent asks "Would you like me to create 'walking'?"
      2. User says "show me my habits" instead of yes/no
      3. Expected: habits are shown, walking is NOT created
    """
    user_id = test_user["id"]

    with (
        patch("app.agent.graph_agent._confirmed", new_callable=AsyncMock, return_value=False),
        patch(
            "app.agent.graph_agent._get_classifier",
            return_value=MagicMock(
                run=AsyncMock(return_value=_make_classifier_result(intent="list"))
            ),
        ),
        patch(
            "app.agent.graph_agent.tools.list_habits",
            new_callable=AsyncMock,
            return_value={"habits": [{"id": 1, "name": "water drinking", "target": 8, "unit": "glasses", "frequency": "daily"}]},
        ),
        patch("app.agent.graph_agent.queries.create_habit", new_callable=AsyncMock) as mock_create,
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent(
            message="show me my habits",
            user_id=user_id,
            awaiting="create_confirm",
            context={"habit_name": "walking"},
        )

    assert response.status == AgentStatus.success
    # Walking was NOT created — the list was shown instead
    mock_create.assert_not_called()
    # No pending confirmation — the user's intent was honoured
    assert response.data is None or response.data.get("awaiting") != "create_confirm"


@pytest.mark.asyncio
async def test_duplicate_confirm_add_creates_second_log(test_user):
    """Replying 'add' to a duplicate-log prompt creates a second log entry for today."""
    user_id = test_user["id"]
    mock_log = AsyncMock(return_value={"log": {"id": 2, "value": 5.0}, "message": "Added"})

    with (
        patch("app.agent.graph_agent._confirmed", new_callable=AsyncMock, return_value=False),
        patch("app.agent.graph_agent.tools.log_activity", mock_log),
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent(
            message="add",
            user_id=user_id,
            awaiting="duplicate_confirm",
            context={"habit_name": "running", "existing_value": 3.0, "new_value": 5.0},
        )

    assert response.status == AgentStatus.success
    mock_log.assert_called_once_with(user_id, "running", 5.0, None)


@pytest.mark.asyncio
async def test_duplicate_confirm_update_replaces_todays_log(test_user):
    """Replying 'update' to a duplicate-log prompt updates today's log instead of adding."""
    user_id = test_user["id"]
    mock_update = AsyncMock(return_value={"log": {"id": 1, "value": 5.0}, "message": "Updated"})

    with (
        patch("app.agent.graph_agent._confirmed", new_callable=AsyncMock, return_value=False),
        patch("app.agent.graph_agent.tools.update_log", mock_update),
        patch("app.agent.graph_agent._get_formatter", _mock_formatter()),
    ):
        response = await run_graph_agent(
            message="update",
            user_id=user_id,
            awaiting="duplicate_confirm",
            context={
                "habit_name": "running",
                "existing_value": 3.0,
                "new_value": 5.0,
                "log_id": 1,
            },
        )

    assert response.status == AgentStatus.success
    mock_update.assert_called_once_with(user_id, 1, 5.0)
