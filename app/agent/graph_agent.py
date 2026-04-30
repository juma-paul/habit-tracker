"""PydanticAI graph agent — deterministic tool sequencing.

Control flow lives in Python (graph edges), not in the LLM.
Two LLM calls per request: ClassifyIntent (structured extraction) and
FormatResponse (friendly prose). All nodes between them are pure Python.

Enabled via CONTROL_MODEL=graph in .env (default: freeform).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import AsyncIterator, cast

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from app.core.config import get_settings
from app.core.logging import log_agent_run
from app.models.schemas import AgentResponse, AgentStatus
from app.agent import tools
from app.db import queries


def _build_model():
    """Return the configured AI model (Anthropic or OpenAI)."""
    settings = get_settings()
    if settings.ai_provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            settings.anthropic_model,
            provider=AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value()
            ),
        )
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        settings.openai_model,
        provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value()),
    )


class IntentResult(BaseModel):
    """Structured output from the intent classifier."""

    intent: str  # log | create | list | progress | delete | update_habit | fix_log | delete_log | other
    habit_name: str | None = None
    new_habit_name: str | None = None
    habit_value: float | None = None
    habit_unit: str | None = None
    habit_target: float | None = None
    habit_frequency: str | None = None
    days: int = 7


@lru_cache
def _get_classifier() -> Agent:
    """Intent-classification agent. No tools; returns IntentResult."""
    return Agent(
        model=_build_model(),
        output_type=IntentResult,
        system_prompt=(
            "You extract the user's intent and habit details from a single message.\n\n"
            "Return exactly one of these intents:\n"
            "- log: user reports doing something ('I ran 5km', 'walked 8000 steps').\n"
            "  Extract habit_name AND habit_value (number only). If no number given, habit_value=null.\n"
            "- create: user explicitly sets up a new habit ('create a running habit').\n"
            "  Extract habit_name, habit_target, habit_unit, habit_frequency.\n"
            "- list: user wants to see their habits ('show my habits', 'what am I tracking?').\n"
            "- progress: user wants stats ('how am I doing with reading?', 'my running stats').\n"
            "  Extract habit_name and days (default 7).\n"
            "- delete: user wants to permanently remove a HABIT ('delete running habit', 'remove reading').\n"
            "  Extract habit_name. If unclear which habit, leave habit_name null.\n"
            "- update_habit: user wants to CHANGE a habit's settings — not a log entry.\n"
            "  Examples: 'change my running goal to 10km', 'rename meditation to mindfulness', 'make reading weekly'.\n"
            "  Extract habit_name (current name), new_habit_name (if renaming), habit_target, habit_unit, habit_frequency.\n"
            "- fix_log: user wants to CORRECT a previous log entry ('I made a mistake', 'only ran 3km not 5', 'fix my log').\n"
            "  Extract habit_name and habit_value (the CORRECT value, not the wrong one).\n"
            "- delete_log: user wants to REMOVE a specific log entry — NOT delete the habit itself.\n"
            "  Examples: 'delete my running log from today', 'remove that last entry', 'undo my last log'.\n"
            "  Extract habit_name.\n"
            "- other: anything else.\n\n"
            "For habit_name, always use the gerund or noun form — never a past-tense verb.\n"
            "Examples: 'I ran' → 'running', 'I meditated' → 'meditation', 'I read' → 'reading', "
            "'I walked' → 'walking', 'I slept 8 hours' → 'sleep'.\n"
            "Key distinction: 'delete running' = delete the habit (intent=delete). "
            "'delete my running log' = delete a log entry (intent=delete_log)."
        ),
    )


@lru_cache
def _get_formatter() -> Agent:
    """Formatting agent. No tools; turns tool output into friendly prose."""
    return Agent(
        model=_build_model(),
        output_type=str,
        system_prompt=(
            "You are a friendly habit tracking assistant. "
            "Given the user's original message and a tool result (JSON dict), "
            "write a short warm response (1–3 sentences). "
            "Rules:\n"
            "- Use the user's exact words for habit names.\n"
            "- Never mention tool names or JSON keys.\n"
            "- Include units with a space: '5 km', '30 min'.\n"
            "- Use commas for large numbers: '8,000 steps'.\n"
            "- For list/progress results, use markdown tables or bullet points.\n"
            "- For confirmations, one sentence is enough: 'Done! Logged 5 km for Running.'\n"
            "- For error results, explain the problem clearly and suggest what to do."
        ),
    )


@dataclass
class HabitGraphState:
    """All data needed to run one user request through the graph."""

    message: str
    user_id: int

    intent: str | None = None
    habit_name: str | None = None
    new_habit_name: str | None = None
    habit_value: float | None = None
    habit_unit: str | None = None
    habit_target: float | None = None
    habit_frequency: str | None = None
    days: int = 7

    habit_exists: bool = False
    log_after_create: bool = False
    log_action: str | None = None

    existing_log_id: int | None = None
    existing_log_value: float | None = None

    tool_result: dict | None = None
    response: str | None = None

    # Multi-turn confirmation state sent by the client:
    # "create_confirm" | "delete_confirm" | "log_value" |
    # "duplicate_confirm" | "fix_log_confirm" | "delete_log_confirm" | None
    awaiting: str | None = None


def _confirmed(message: str) -> bool:
    """Return True if the user's message is an affirmative reply."""
    return message.strip().lower() in {
        "yes",
        "y",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "create it",
        "do it",
        "go ahead",
        "confirm",
    }


@dataclass
class ClassifyIntent(BaseNode[HabitGraphState]):
    """Entry node — routes to HandleConfirmation if awaiting, else classifies intent."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> (
        CheckHabitExists
        | AskForValueNode
        | AskCreateConfirmation
        | ListHabitsNode
        | GetProgressNode
        | CreateHabitNode
        | AskDeleteConfirmation
        | UpdateHabitNode
        | FetchRecentLogsNode
        | HandleConfirmation
        | End[AgentResponse]
    ):
        state = ctx.state

        if state.awaiting:
            logger.debug(f"Graph: awaiting={state.awaiting!r} → HandleConfirmation")
            return HandleConfirmation()

        logger.debug(f"Graph: classifying {state.message[:50]!r}")
        result = await _get_classifier().run(state.message)
        intent = cast(IntentResult, result.output)

        state.intent = intent.intent
        state.habit_name = intent.habit_name
        state.new_habit_name = intent.new_habit_name
        state.habit_value = intent.habit_value
        state.habit_unit = intent.habit_unit
        state.habit_target = intent.habit_target
        state.habit_frequency = intent.habit_frequency
        state.days = intent.days

        logger.debug(
            f"Graph: intent={state.intent!r} habit={state.habit_name!r} value={state.habit_value!r}"
        )

        if intent.intent == "log":
            if state.habit_value is None:
                return AskForValueNode()
            return CheckHabitExists()

        elif intent.intent == "list":
            return ListHabitsNode()

        elif intent.intent == "progress":
            return GetProgressNode()

        elif intent.intent == "create":
            return AskCreateConfirmation()

        elif intent.intent == "delete":
            if not state.habit_name:
                state.response = "Which habit would you like to delete?"
                return End(
                    AgentResponse(status=AgentStatus.success, message=state.response)
                )
            return AskDeleteConfirmation()

        elif intent.intent == "update_habit":
            return UpdateHabitNode()

        elif intent.intent == "fix_log":
            state.log_action = "fix"
            return FetchRecentLogsNode()

        elif intent.intent == "delete_log":
            state.log_action = "delete_log"
            return FetchRecentLogsNode()

        else:
            state.response = (
                "I can help you track habits! Try:\n"
                "- 'I ran 5 km'\n"
                "- 'show my habits'\n"
                "- 'how am I doing with reading this week?'\n"
                "- 'change my running goal to 10 km'\n"
                "- 'I made a mistake, I only ran 3 km'"
            )
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )


@dataclass
class HandleConfirmation(BaseNode[HabitGraphState]):
    """Dispatches the user's reply based on state.awaiting."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> (
        CheckHabitExists
        | CreateHabitNode
        | DeleteHabitNode
        | LogActivityNode
        | UpdateTodayLogNode
        | UpdateLogNode
        | DeleteLogNode
        | End[AgentResponse]
    ):
        state = ctx.state
        confirmed = _confirmed(state.message)

        if state.awaiting == "create_confirm":
            state.awaiting = None
            if confirmed:
                state.log_after_create = state.habit_value is not None
                return CreateHabitNode()
            state.response = f"No problem! I won't create '{state.habit_name}'."
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )

        if state.awaiting == "delete_confirm":
            state.awaiting = None
            if confirmed:
                return DeleteHabitNode()
            state.response = f"Got it — '{state.habit_name}' is safe."
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )

        if state.awaiting == "log_value":
            match = re.search(r"\d+\.?\d*", state.message)
            if match:
                state.habit_value = float(match.group())
                state.awaiting = None
                return CheckHabitExists()
            state.response = (
                "I didn't catch a number. How much did you do? (e.g. '30', '5.5')"
            )
            return End(
                AgentResponse(
                    status=AgentStatus.clarification,
                    message=state.response,
                    data={"awaiting": "log_value", "habit_name": state.habit_name},
                )
            )

        if state.awaiting == "duplicate_confirm":
            state.awaiting = None
            msg = state.message.strip().lower()
            if any(w in msg for w in ("add", "another", "new", "both", "second")):
                return LogActivityNode()
            elif any(w in msg for w in ("update", "change", "replace", "fix", "edit")):
                return UpdateTodayLogNode()
            else:
                state.response = "OK, nothing was changed."
                return End(
                    AgentResponse(status=AgentStatus.success, message=state.response)
                )

        if state.awaiting == "fix_log_confirm":
            state.awaiting = None
            if confirmed:
                return UpdateLogNode()
            state.response = "OK, I'll leave your log as is."
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )

        if state.awaiting == "delete_log_confirm":
            state.awaiting = None
            if confirmed:
                return DeleteLogNode()
            state.response = "Got it — log entry kept."
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )

        state.response = (
            "I'm not sure what you're confirming. What would you like to do?"
        )
        return End(AgentResponse(status=AgentStatus.success, message=state.response))


@dataclass
class AskForValueNode(BaseNode[HabitGraphState]):
    """Ask for a numeric value when the user reports an activity without one."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        state.response = (
            f"How much {state.habit_name} did you do? "
            "Reply with just the number (e.g. '30', '5.5')."
        )
        return End(
            AgentResponse(
                status=AgentStatus.success,
                message=state.response,
                data={"awaiting": "log_value", "habit_name": state.habit_name},
            )
        )


@dataclass
class CheckHabitExists(BaseNode[HabitGraphState]):
    """Check if the named habit exists; route to duplicate check or create confirmation."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> CheckDuplicateTodayNode | AskCreateConfirmation:
        state = ctx.state
        result = await tools.list_habits(state.user_id)
        habits = result.get("habits", [])
        name_lower = (state.habit_name or "").lower()
        state.habit_exists = any(name_lower in h["name"].lower() for h in habits)

        if state.habit_exists:
            return CheckDuplicateTodayNode()
        return AskCreateConfirmation()


@dataclass
class CheckDuplicateTodayNode(BaseNode[HabitGraphState]):
    """Check for a same-day log; ask add-vs-update or proceed to logging."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> LogActivityNode | AskDuplicateConfirmation:
        state = ctx.state
        assert state.habit_name is not None
        habit = await queries.find_habit_by_name(state.user_id, state.habit_name)
        if not habit:
            # Shouldn't happen (CheckHabitExists passed), but safe to fall through.
            return LogActivityNode()

        today_logs = await queries.get_today_logs(habit["id"], state.user_id)
        if today_logs:
            state.existing_log_id = today_logs[0]["id"]
            state.existing_log_value = today_logs[0]["value"]
            return AskDuplicateConfirmation()

        return LogActivityNode()


@dataclass
class AskDuplicateConfirmation(BaseNode[HabitGraphState]):
    """Prompt when a same-day log already exists — add another or update."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        state.response = (
            f"You already logged {state.existing_log_value} for '{state.habit_name}' today. "
            f"Do you want to add another entry of {state.habit_value}, "
            f"or update today's log to {state.habit_value}?\n"
            "Reply 'add' for a new entry, or 'update' to replace today's log."
        )
        return End(
            AgentResponse(
                status=AgentStatus.success,
                message=state.response,
                data={
                    "awaiting": "duplicate_confirm",
                    "habit_name": state.habit_name,
                    "existing_value": state.existing_log_value,
                    "new_value": state.habit_value,
                },
            )
        )


@dataclass
class AskCreateConfirmation(BaseNode[HabitGraphState]):
    """Prompt before creating a habit that doesn't exist yet."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        state.response = (
            f"I don't have a habit called '{state.habit_name}' yet. "
            "Would you like me to create it?"
        )
        return End(
            AgentResponse(
                status=AgentStatus.success,
                message=state.response,
                data={
                    "awaiting": "create_confirm",
                    "habit_name": state.habit_name,
                    "habit_value": state.habit_value,
                },
            )
        )


@dataclass
class AskDeleteConfirmation(BaseNode[HabitGraphState]):
    """Prompt before permanently deleting a habit."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        state.response = (
            f"Are you sure you want to permanently delete '{state.habit_name}'? "
            "This cannot be undone. Reply 'yes' to confirm."
        )
        return End(
            AgentResponse(
                status=AgentStatus.success,
                message=state.response,
                data={"awaiting": "delete_confirm", "habit_name": state.habit_name},
            )
        )


@dataclass
class CreateHabitNode(BaseNode[HabitGraphState]):
    """Create a habit; continue to LogActivityNode if log_after_create is set."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> LogActivityNode | FormatResponse:
        state = ctx.state
        assert state.habit_name is not None
        result = await tools.create_habit(
            state.user_id,
            state.habit_name,
            state.habit_target,
            state.habit_unit,
            state.habit_frequency or "daily",
        )
        state.tool_result = result
        if state.log_after_create:
            state.log_after_create = False
            return LogActivityNode()
        return FormatResponse()


@dataclass
class LogActivityNode(BaseNode[HabitGraphState]):
    """Log a value for an existing habit."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        assert state.habit_name is not None
        assert state.habit_value is not None
        result = await tools.log_activity(
            state.user_id,
            state.habit_name,
            state.habit_value,
            None,
        )
        state.tool_result = result
        return FormatResponse()


@dataclass
class UpdateTodayLogNode(BaseNode[HabitGraphState]):
    """Update today's existing log (from duplicate detection flow)."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        assert state.existing_log_id is not None
        result = await tools.update_log(
            state.user_id,
            state.existing_log_id,
            state.habit_value,
        )
        state.tool_result = result
        return FormatResponse()


@dataclass
class ListHabitsNode(BaseNode[HabitGraphState]):
    """List all active habits for the user."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        state.tool_result = await tools.list_habits(state.user_id)
        return FormatResponse()


@dataclass
class GetProgressNode(BaseNode[HabitGraphState]):
    """Get progress statistics for a habit over `days` days."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        assert state.habit_name is not None
        state.tool_result = await tools.get_progress(
            state.user_id, state.habit_name, state.days
        )
        return FormatResponse()


@dataclass
class UpdateHabitNode(BaseNode[HabitGraphState]):
    """Update a habit's settings (name, target, unit, frequency)."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        assert state.habit_name is not None
        result = await tools.update_habit(
            state.user_id,
            state.habit_name,
            state.new_habit_name,
            state.habit_target,
            state.habit_unit,
            state.habit_frequency,
        )
        state.tool_result = result
        return FormatResponse()


@dataclass
class DeleteHabitNode(BaseNode[HabitGraphState]):
    """Delete a habit after the user has confirmed."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        assert state.habit_name is not None
        state.tool_result = await tools.delete_habit(state.user_id, state.habit_name)
        return FormatResponse()


@dataclass
class FetchRecentLogsNode(BaseNode[HabitGraphState]):
    """Find the most recent log entry for fix_log and delete_log flows.

    Priority: today's log → most recent within 7 days → not found.
    """

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> AskFixLogConfirmation | AskDeleteLogConfirmation | End[AgentResponse]:
        state = ctx.state
        assert state.habit_name is not None

        habit = await queries.find_habit_by_name(state.user_id, state.habit_name)
        if not habit:
            state.response = f"I couldn't find a habit called '{state.habit_name}'."
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )

        today_logs = await queries.get_today_logs(habit["id"], state.user_id)
        if today_logs:
            state.existing_log_id = today_logs[0]["id"]
            state.existing_log_value = today_logs[0]["value"]
        else:
            recent = await queries.get_logs(habit["id"], state.user_id, days=7)
            if recent:
                state.existing_log_id = recent[0]["id"]
                state.existing_log_value = recent[0]["value"]
            else:
                state.response = f"You haven't logged '{state.habit_name}' recently — nothing to change."
                return End(
                    AgentResponse(status=AgentStatus.success, message=state.response)
                )

        if state.log_action == "fix":
            return AskFixLogConfirmation()
        return AskDeleteLogConfirmation()


@dataclass
class AskFixLogConfirmation(BaseNode[HabitGraphState]):
    """Ask user to confirm correcting a log entry."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        state.response = (
            f"I found your '{state.habit_name}' log: {state.existing_log_value}. "
            f"Update it to {state.habit_value}? Reply 'yes' to confirm."
        )
        return End(
            AgentResponse(
                status=AgentStatus.success,
                message=state.response,
                data={
                    "awaiting": "fix_log_confirm",
                    "log_id": state.existing_log_id,
                    "current_value": state.existing_log_value,
                    "new_value": state.habit_value,
                },
            )
        )


@dataclass
class AskDeleteLogConfirmation(BaseNode[HabitGraphState]):
    """Ask user to confirm deleting a log entry (not the habit itself)."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        state.response = (
            f"Delete your '{state.habit_name}' log entry ({state.existing_log_value})? "
            "This cannot be undone. Reply 'yes' to confirm."
        )
        return End(
            AgentResponse(
                status=AgentStatus.success,
                message=state.response,
                data={
                    "awaiting": "delete_log_confirm",
                    "log_id": state.existing_log_id,
                    "value": state.existing_log_value,
                },
            )
        )


@dataclass
class UpdateLogNode(BaseNode[HabitGraphState]):
    """Update a log entry after fix_log confirmation."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        assert state.existing_log_id is not None
        result = await tools.update_log(
            state.user_id,
            state.existing_log_id,
            state.habit_value,
        )
        state.tool_result = result
        return FormatResponse()


@dataclass
class DeleteLogNode(BaseNode[HabitGraphState]):
    """Delete a log entry after delete_log confirmation."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        assert state.existing_log_id is not None
        result = await tools.delete_log(state.user_id, state.existing_log_id)
        state.tool_result = result
        return FormatResponse()


@dataclass
class FormatResponse(BaseNode[HabitGraphState]):
    """LLM call #2 — turn tool_result into a user-facing message."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        prompt = (
            f"User said: '{state.message}'\n"
            f"Tool result: {state.tool_result}\n"
            "Write a short, friendly response."
        )
        result = await _get_formatter().run(prompt)
        state.response = result.output
        return End(AgentResponse(status=AgentStatus.success, message=state.response))


habit_graph = Graph(
    nodes=(
        ClassifyIntent,
        HandleConfirmation,
        AskForValueNode,
        CheckHabitExists,
        CheckDuplicateTodayNode,
        AskDuplicateConfirmation,
        AskCreateConfirmation,
        LogActivityNode,
        UpdateTodayLogNode,
        CreateHabitNode,
        ListHabitsNode,
        GetProgressNode,
        UpdateHabitNode,
        AskDeleteConfirmation,
        DeleteHabitNode,
        FetchRecentLogsNode,
        AskFixLogConfirmation,
        AskDeleteLogConfirmation,
        UpdateLogNode,
        DeleteLogNode,
        FormatResponse,
    )
)


async def run_graph_agent(
    message: str,
    user_id: int,
    conversation_id: int | None = None,
    awaiting: str | None = None,
    context: dict | None = None,
) -> AgentResponse:
    """Run the full graph and return a complete AgentResponse."""
    start = perf_counter()
    state = HabitGraphState(message=message, user_id=user_id, awaiting=awaiting)

    # Restore stateful fields from the previous response's data payload.
    # The server is stateless; the client echoes context back so we can
    # reconstruct habit_name, log IDs, etc. before re-entering the graph.
    if awaiting and context:
        state.habit_name = context.get("habit_name")
        state.habit_value = context.get("habit_value") or context.get("new_value")
        state.existing_log_id = context.get("log_id")
        state.existing_log_value = context.get("existing_value") or context.get("value")

    if conversation_id:
        await queries.add_message(conversation_id, "user", message)

    result = await habit_graph.run(start_node=ClassifyIntent(), state=state)
    response = cast(AgentResponse, result.output)
    elapsed_ms = (perf_counter() - start) * 1000

    if conversation_id and response.message:
        await queries.add_message(conversation_id, "assistant", response.message)

    log_agent_run(
        user_id=user_id,
        message=message,
        response=response.message,
        elapsed_ms=elapsed_ms,
    )
    return response


async def run_graph_stream(
    message: str,
    user_id: int,
    conversation_id: int | None = None,
    awaiting: str | None = None,
    context: dict | None = None,
) -> AsyncIterator[str]:
    """Run the graph and yield the final message as a single chunk.

    True token-by-token streaming inside FormatResponse is a future enhancement.
    """
    response = await run_graph_agent(message, user_id, conversation_id, awaiting, context)
    if response.message:
        yield response.message
