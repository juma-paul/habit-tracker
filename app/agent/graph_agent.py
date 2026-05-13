"""PydanticAI graph agent — deterministic tool sequencing.

Control flow lives in Python (graph edges), not in the LLM.
Two LLM calls per request: ClassifyIntent (structured extraction) and
FormatResponse (friendly prose). All nodes between them are pure Python.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import cast

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from app.agent import tools
from app.core.config import get_settings
from app.core.logging import log_agent_run
from app.db import queries
from app.models.schemas import AgentResponse, AgentStatus


def _build_model(*, fast: bool = False):
    """Return the configured LLM.

    fast=True  → Haiku / gpt-4o-mini  (structured extraction)
    fast=False → configured model      (response formatting)
    """
    settings = get_settings()
    if settings.ai_provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        assert settings.anthropic_api_key is not None, "ANTHROPIC_API_KEY must be set when AI_PROVIDER=anthropic"
        return AnthropicModel(
            "claude-haiku-4-5-20251001" if fast else settings.anthropic_model,
            provider=AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value()
            ),
        )
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    assert settings.openai_api_key is not None, "OPENAI_API_KEY must be set when AI_PROVIDER=openai"
    return OpenAIChatModel(
        "gpt-4o-mini" if fast else settings.openai_model,
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
    """Intent-classification agent. Uses fast model — simple extraction, no quality needed."""
    return Agent(
        model=_build_model(fast=True),
        output_type=IntentResult,
        system_prompt=(
            "You extract the user's intent and habit details from a single message.\n\n"
            "Return exactly one of these intents:\n"
            "- log: user reports doing something ('I ran 5km', 'walked 8000 steps').\n"
            "  Extract habit_name AND habit_value (number only). If no number given, habit_value=null.\n"
            "- create: user explicitly sets up a new habit ('create a running habit', 'add water drinking').\n"
            "  Extract habit_name, habit_target, habit_unit, habit_frequency.\n"
            "- list: user wants to see their habits ('show my habits', 'what am I tracking?', 'list my habits',\n"
            "  'what habits do I have?', 'show me my habits'). Use ONLY when asking for the habit definitions.\n"
            "- progress: user wants performance data or stats — for one habit OR all habits.\n"
            "  Examples: 'how am I doing with reading?', 'my running stats', 'show my overall progress',\n"
            "  'how is my progress', 'all my progress', 'overall stats', 'how have I been doing'.\n"
            "  Extract habit_name (null if asking about ALL habits) and days (default 7).\n"
            "- delete: user wants to permanently remove a HABIT (not a log entry).\n"
            "  Examples: 'delete running habit', 'remove reading', 'remove the water drinking habit',\n"
            "  'delete the one without a target', 'remove the duplicate'.\n"
            "  Use delete even for vague references like 'remove it' or 'delete that one' when context\n"
            "  suggests a habit (not a log). Extract habit_name if identifiable; null if unclear.\n"
            "- update_habit: user wants to CHANGE a habit's settings — not a log entry.\n"
            "  Examples: 'change my running goal to 10km', 'rename meditation to mindfulness',\n"
            "  'update the target to 12 glasses', 'set goal to 8000 steps', 'make reading weekly'.\n"
            "  Extract habit_name (current name), new_habit_name (if renaming), habit_target, habit_unit,\n"
            "  habit_frequency. For 'update target to 12 glasses': habit_target=12, habit_unit='glasses'.\n"
            "- fix_log: user wants to CORRECT a previous log entry ('I made a mistake', 'only ran 3km not 5').\n"
            "  Extract habit_name and habit_value (the CORRECT value).\n"
            "- delete_log: user wants to REMOVE a specific LOG ENTRY — NOT the habit itself.\n"
            "  Use ONLY when the user explicitly mentions log, entry, record, or undo:\n"
            "  'delete my running log from today', 'remove that log entry', 'undo my last log',\n"
            "  'delete the log', 'remove that entry'. Never use for habit deletion requests.\n"
            "- other: greetings, thanks, small talk, or anything that isn't a habit-tracking action.\n\n"
            "CRITICAL DISTINCTION — delete vs delete_log:\n"
            "  delete      = removing the habit itself permanently ('delete running', 'remove it',\n"
            "                'yes remove it', 'remove the duplicate', 'remove water drinking')\n"
            "  delete_log  = removing a log ENTRY ('delete that log', 'undo my entry', 'remove the record')\n"
            "  Default to delete when the message is ambiguous ('remove it', 'yes delete it').\n\n"
            "For habit_name, always use the gerund or noun form — never a past-tense verb.\n"
            "Examples: 'I ran' → 'running', 'I meditated' → 'meditation', 'I read' → 'reading',\n"
            "'water drinking' → 'water drinking' (keep full compound nouns intact, do not truncate).\n"
            "Extract the COMPLETE habit name, e.g. 'water drinking' not just 'water'."
        ),
    )


@lru_cache
def _get_confirmation_agent() -> Agent:
    """Single-purpose agent: returns True if the user is agreeing, False if declining."""
    return Agent(
        model=_build_model(fast=True),
        output_type=bool,
        system_prompt=(
            "You decide whether a user's reply is an affirmative or negative response "
            "to a yes/no confirmation prompt.\n"
            "Return TRUE for any agreement, including:\n"
            "  - Direct yes: yes, yeah, yep, yup, sure, ok, okay, correct, right, absolutely\n"
            "  - Imperative commands that mean proceed: 'do it', 'go ahead', 'proceed', "
            "'create it', 'delete it', 'confirm', 'make it', 'add it', 'yes create', 'yes delete'\n"
            "  - Foreign language: sim (Portuguese), ja (German), oui (French), sí (Spanish)\n"
            "  - Positive phrases: 'sounds good', 'let's do it', 'please do', 'go for it'\n"
            "Return FALSE for any refusal: no, nope, cancel, stop, don't, nevermind, skip, etc.\n"
            "When truly ambiguous (not clearly yes OR no), return false."
        ),
    )


@lru_cache
def _get_formatter() -> Agent:
    """Formatting agent. No tools; turns tool output into friendly prose."""
    return Agent(
        model=_build_model(),
        output_type=str,  # fast=False → full-quality formatter
        system_prompt=(
            "You are a friendly habit tracking assistant. "
            "Given the user's original message and a tool result (JSON dict), "
            "write a clear, warm response. "
            "Rules:\n"
            "- ALWAYS respond in English, regardless of the user's language or device locale.\n"
            "- Use the user's exact words for habit names.\n"
            "- Never mention tool names or JSON keys.\n"
            "- Always write units as full words: '5 kilometers', '30 minutes', '10 kilograms', "
            "'8,000 steps', '2 hours', '500 grams'. Never use abbreviations (km, min, kg, g, etc.).\n"
            "- Use commas for large numbers: '8,000 steps'.\n"
            "- ALWAYS use a markdown table when showing lists of habits, logs, or progress. "
            "  Table columns for habit lists: Habit | Target | Frequency | Streak. "
            "  Table columns for logs/progress: Date | Habit | Value | vs. Target. "
            "  Use | to separate columns and include a separator row (|---|---|). "
            "  Never use plain bullet points for structured data.\n"
            "- For confirmations, one sentence is enough: 'Done! Logged 5 km for Running.'\n"
            "- For error results, explain the problem clearly and suggest what to do.\n"
            "- Keep prose short (1-2 sentences) before/after tables."
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

    # Multiple log entries offered for selection during delete_log flow
    candidate_logs: list[dict] | None = None

    # Set during deletion disambiguation when user has two habits with identical names
    habit_id: int | None = None    # the specific habit to delete (by DB id)
    habit_ids: list[int] | None = None  # candidates shown in the numbered list

    tool_result: dict | None = None
    response: str | None = None

    # When set, the streaming path should stream this prompt through _get_formatter()
    # instead of using the pre-baked response.  Set by FormatResponse.
    format_prompt: str | None = None

    # Recent conversation history injected by _run_graph_and_get_state for context
    conversation_history: str | None = None

    # Multi-turn confirmation state sent by the client:
    # "create_confirm" | "delete_confirm" | "log_value" |
    # "duplicate_confirm" | "fix_log_confirm" | "delete_log_confirm" | None
    awaiting: str | None = None


async def _confirmed(message: str) -> bool:
    """Return True if the user's message is an affirmative reply.

    Uses the LLM so any natural language confirmation works — 'Yeah!',
    'sure thing', 'sim' (Portuguese), 'ja' (German), etc.
    """
    result = await _get_confirmation_agent().run(message)
    return cast(bool, result.output)


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
        classify_input = state.message
        if state.conversation_history:
            classify_input = (
                f"Recent conversation context:\n{state.conversation_history}\n\n"
                f"Current message to classify: {state.message}"
            )
        result = await _get_classifier().run(classify_input)
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

        match intent.intent:
            case "log":
                if state.habit_value is None:
                    return AskForValueNode()
                return CheckHabitExists()

            case "list":
                return ListHabitsNode()

            case "progress":
                return GetProgressNode()

            case "create":
                # Prevent duplicates — check if a habit with this name already exists
                if state.habit_name:
                    existing_result = await tools.list_habits(state.user_id)
                    name_lower = state.habit_name.lower()
                    existing = next(
                        (
                            h
                            for h in existing_result.get("habits", [])
                            if name_lower in h["name"].lower()
                            or h["name"].lower() in name_lower
                        ),
                        None,
                    )
                    if existing:
                        target_info = (
                            f" with a target of {existing['target']} {existing.get('unit', '')}".rstrip()
                            if existing.get("target")
                            else ""
                        )
                        state.response = (
                            f"You already have a '{existing['name']}' habit{target_info}. "
                            "Would you like to update its settings instead?"
                        )
                        return End(
                            AgentResponse(
                                status=AgentStatus.success, message=state.response
                            )
                        )
                # User explicitly said "create" — no confirmation needed
                return CreateHabitNode()

            case "delete":
                if not state.habit_name:
                    # Show the user's habits so they can name one precisely
                    habits_result = await tools.list_habits(state.user_id)
                    habits = habits_result.get("habits", [])
                    if habits:
                        names = ", ".join(f"'{h['name']}'" for h in habits)
                        state.response = (
                            f"Which habit would you like to delete? Your habits: {names}."
                        )
                    else:
                        state.response = "You don't have any habits to delete yet."
                    return End(
                        AgentResponse(status=AgentStatus.success, message=state.response)
                    )
                return AskDeleteConfirmation()

            case "update_habit":
                return UpdateHabitNode()

            case "fix_log":
                state.log_action = "fix"
                return FetchRecentLogsNode()

            case "delete_log":
                state.log_action = "delete_log"
                return FetchRecentLogsNode()

            case _:
                # Conversational, social, or genuinely unclear — let the formatter respond
                # naturally rather than returning a hardcoded suggestions list.
                state.format_prompt = (
                    f"User said: '{state.message}'\n"
                    "This is a conversational message — a greeting, thanks, small talk, "
                    "or something that doesn't map to a habit tracking action. "
                    "Respond warmly and naturally in 1-2 sentences. "
                    "If it's thanks or a compliment, accept gracefully. "
                    "If it's truly unclear what they want, gently mention you help with "
                    "habit tracking — but do NOT list commands or use bullet points."
                )
                return End(AgentResponse(status=AgentStatus.success, message=""))


@dataclass
class HandleConfirmation(BaseNode[HabitGraphState]):
    """Dispatches the user's reply based on state.awaiting."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> (
        CheckHabitExists
        | AskCreateConfirmation
        | AskForValueNode
        | CreateHabitNode
        | DeleteHabitNode
        | ListHabitsNode
        | GetProgressNode
        | AskDeleteConfirmation
        | FetchRecentLogsNode
        | LogActivityNode
        | UpdateHabitNode
        | UpdateTodayLogNode
        | UpdateLogNode
        | DeleteLogNode
        | End[AgentResponse]
    ):
        state = ctx.state
        confirmed = await _confirmed(state.message)

        match state.awaiting:
            case "create_confirm":
                state.awaiting = None
                # Always re-classify — user may include target/unit/frequency in their
                # confirmation ("yes, with a goal of 12 glasses daily") or send a
                # completely different request ("show my habits") that should be honoured.
                result = await _get_classifier().run(state.message)
                intent = cast(IntentResult, result.output)

                if confirmed:
                    # Merge any details the user added in their confirmation message
                    state.habit_name = intent.habit_name or state.habit_name
                    if intent.habit_target is not None:
                        state.habit_target = intent.habit_target
                    if intent.habit_unit is not None:
                        state.habit_unit = intent.habit_unit
                    if intent.habit_frequency is not None:
                        state.habit_frequency = intent.habit_frequency
                    if intent.habit_value is not None:
                        state.habit_value = intent.habit_value
                    state.log_after_create = state.habit_value is not None
                    return CreateHabitNode()

                # Not a confirmation — if it's a different valid intent, honour it.
                match intent.intent:
                    case "create":
                        state.habit_name = intent.habit_name or state.habit_name
                        state.habit_target = intent.habit_target
                        state.habit_unit = intent.habit_unit
                        state.habit_frequency = intent.habit_frequency
                        state.habit_value = intent.habit_value
                        state.log_after_create = state.habit_value is not None
                        return CreateHabitNode()
                    case "list":
                        return ListHabitsNode()
                    case "progress":
                        state.habit_name = intent.habit_name
                        state.days = intent.days
                        return GetProgressNode()
                    case "log":
                        state.habit_name = intent.habit_name or state.habit_name
                        state.habit_value = intent.habit_value
                        if state.habit_value is None:
                            return AskForValueNode()
                        return CheckHabitExists()
                    case "delete":
                        state.habit_name = intent.habit_name
                        if not state.habit_name:
                            state.response = "Which habit would you like to delete?"
                            return End(
                                AgentResponse(
                                    status=AgentStatus.success, message=state.response
                                )
                            )
                        return AskDeleteConfirmation()
                    case "update_habit":
                        state.habit_name = intent.habit_name or state.habit_name
                        state.new_habit_name = intent.new_habit_name
                        state.habit_target = intent.habit_target
                        state.habit_unit = intent.habit_unit
                        state.habit_frequency = intent.habit_frequency
                        return UpdateHabitNode()
                    case _:
                        # Genuinely a cancel / unrecognised
                        state.response = f"No problem! I won't create '{state.habit_name}'."
                        return End(
                            AgentResponse(status=AgentStatus.success, message=state.response)
                        )

            case "delete_disambiguate":
                state.awaiting = None
                ids = state.habit_ids or []
                if not ids:
                    state.response = (
                        "Sorry, I lost track. Please say 'delete [habit name]' to try again."
                    )
                    return End(
                        AgentResponse(status=AgentStatus.success, message=state.response)
                    )

                # Parse the user's choice — try number first, then ordinal words
                msg_lower = state.message.strip().lower()
                index: int | None = None

                num_match = re.search(r"\b([1-9])\b", state.message)
                if num_match:
                    n = int(num_match.group(1))
                    if 1 <= n <= len(ids):
                        index = n - 1

                if index is None:
                    # Named words checked before number-words so "second one" matches
                    # "second" (index 1) rather than "one" (index 0).
                    # Word boundaries prevent "one" from matching inside "someone".
                    ordinals = {
                        "first": 0, "second": 1, "third": 2,
                        "one": 0, "two": 1, "three": 2,
                    }
                    for word, idx in ordinals.items():
                        if re.search(rf"\b{word}\b", msg_lower) and idx < len(ids):
                            index = idx
                            break

                if index is None:
                    # Couldn't parse — re-prompt
                    state.response = (
                        f"Please reply with a number (1–{len(ids)})."
                    )
                    return End(
                        AgentResponse(
                            status=AgentStatus.success,
                            message=state.response,
                            data={
                                "awaiting": "delete_disambiguate",
                                "habit_ids": ids,
                                "habit_name": state.habit_name,
                            },
                        )
                    )

                state.habit_id = ids[index]
                # Fetch the selected habit for a labelled confirmation prompt
                chosen = await queries.get_habit(state.habit_id, state.user_id)
                label = _habit_label(dict(chosen)) if chosen else ""
                display = (
                    f"'{state.habit_name}' ({label})"
                    if label
                    else f"'{state.habit_name}'"
                )
                state.response = (
                    f"Delete {display}? This cannot be undone. Reply 'yes' to confirm."
                )
                return End(
                    AgentResponse(
                        status=AgentStatus.success,
                        message=state.response,
                        data={
                            "awaiting": "delete_confirm",
                            "habit_id": state.habit_id,
                            "habit_name": state.habit_name,
                        },
                    )
                )

            case "delete_confirm":
                state.awaiting = None
                if confirmed:
                    return DeleteHabitNode()
                state.response = f"Got it — '{state.habit_name}' is safe."
                return End(
                    AgentResponse(status=AgentStatus.success, message=state.response)
                )

            case "log_value":
                num_match = re.search(r"\d+\.?\d*", state.message)
                if num_match:
                    state.habit_value = float(num_match.group())
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

            case "duplicate_confirm":
                state.awaiting = None
                msg_lower = state.message.strip().lower()
                if confirmed or any(
                    w in msg_lower for w in ("add", "another", "new", "both", "second")
                ):
                    return LogActivityNode()
                elif any(
                    w in msg_lower
                    for w in ("update", "change", "replace", "fix", "edit", "today")
                ):
                    return UpdateTodayLogNode()
                else:
                    state.response = "OK, nothing was changed."
                    return End(
                        AgentResponse(status=AgentStatus.success, message=state.response)
                    )

            case "fix_log_confirm":
                state.awaiting = None
                if confirmed:
                    return UpdateLogNode()
                state.response = "OK, I'll leave your log as is."
                return End(
                    AgentResponse(status=AgentStatus.success, message=state.response)
                )

            case "delete_log_select":
                state.awaiting = None
                logs = state.candidate_logs or []
                msg_lower = state.message.strip().lower()

                # Try to parse a number
                log_idx: int | None = None
                try:
                    log_idx = int(msg_lower) - 1
                except ValueError:
                    ordinals = {
                        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
                        "one": 0, "two": 1, "three": 2, "four": 3, "five": 4,
                    }
                    for word, pos in ordinals.items():
                        if re.search(rf"\b{word}\b", msg_lower):
                            log_idx = pos
                            break

                if log_idx is None or log_idx < 0 or log_idx >= len(logs):
                    state.response = f"Please reply with a number between 1 and {len(logs)}."
                    return End(
                        AgentResponse(
                            status=AgentStatus.success,
                            message=state.response,
                            data={
                                "awaiting": "delete_log_select",
                                "candidate_logs": logs,
                                "habit_name": state.habit_name,
                            },
                        )
                    )

                chosen = logs[log_idx]
                state.existing_log_id = chosen["id"]
                state.existing_log_value = chosen["value"]
                date_str = str(chosen["logged_at"])[:10]
                state.response = (
                    f"Delete log #{log_idx + 1} ({chosen['value']} on {date_str})? "
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
                            "habit_name": state.habit_name,
                        },
                    )
                )

            case "delete_log_confirm":
                state.awaiting = None
                if confirmed:
                    return DeleteLogNode()
                state.response = "Got it — log entry kept."
                return End(
                    AgentResponse(status=AgentStatus.success, message=state.response)
                )

            case _:
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
    """Prompt before permanently deleting a habit.

    If multiple habits match the name, list them so the user can be precise.
    """

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        assert state.habit_name is not None

        # Check for ambiguous match — multiple habits share the same substring
        habits_result = await tools.list_habits(state.user_id)
        name_lower = state.habit_name.lower()
        matches = [
            h
            for h in habits_result.get("habits", [])
            if name_lower in h["name"].lower() or h["name"].lower() in name_lower
        ]

        if len(matches) > 1:
            # All names identical (e.g. two "water drinking" habits) — numbered list
            if len({h["name"] for h in matches}) == 1:
                options = "\n".join(
                    f"  {i + 1}. {_habit_label(h)}" for i, h in enumerate(matches)
                )
                state.response = (
                    f"I found {len(matches)} '{state.habit_name}' habits:\n\n"
                    f"{options}\n\n"
                    "Which one to delete? Reply with the number."
                )
                return End(
                    AgentResponse(
                        status=AgentStatus.success,
                        message=state.response,
                        data={
                            "awaiting": "delete_disambiguate",
                            "habit_ids": [h["id"] for h in matches],
                            "habit_name": state.habit_name,
                        },
                    )
                )
            # Different names but overlapping — ask for the exact name
            names = ", ".join(f"'{h['name']}'" for h in matches)
            state.response = (
                f"I found multiple habits matching '{state.habit_name}': {names}. "
                "Which one would you like to delete? Please use the exact name."
            )
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )

        if len(matches) == 1:
            # Lock onto the exact name so deletion can't miss
            state.habit_name = matches[0]["name"]

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
    """Get progress statistics for one or all habits."""

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        if state.habit_name:
            state.tool_result = await tools.get_progress(
                state.user_id, state.habit_name, state.days
            )
        else:
            # "show my overall progress" — aggregate across all habits
            all_habits = await tools.list_habits(state.user_id)
            results = []
            for habit in all_habits.get("habits", []):
                progress = await tools.get_progress(
                    state.user_id, habit["name"], state.days
                )
                results.append(progress)
            state.tool_result = {"all_progress": results, "days": state.days}
        return FormatResponse()


@dataclass
class UpdateHabitNode(BaseNode[HabitGraphState]):
    """Update a habit's settings (name, target, unit, frequency)."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> FormatResponse | End[AgentResponse]:
        state = ctx.state
        assert state.habit_name is not None

        # Guard: verify habit exists before updating
        habits_result = await tools.list_habits(state.user_id)
        name_lower = state.habit_name.lower()
        matches = [
            h
            for h in habits_result.get("habits", [])
            if name_lower in h["name"].lower() or h["name"].lower() in name_lower
        ]
        if not matches:
            habits = habits_result.get("habits", [])
            if habits:
                names = ", ".join(f"'{h['name']}'" for h in habits)
                state.response = (
                    f"I couldn't find '{state.habit_name}'. Your habits are: {names}."
                )
            else:
                state.response = "You don't have any habits yet. Create one first!"
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )
        if len(matches) > 1:
            names = ", ".join(f"'{h['name']}'" for h in matches)
            state.response = (
                f"I found multiple habits matching '{state.habit_name}': {names}. "
                "Which one would you like to update? Please use the exact name."
            )
            return End(
                AgentResponse(status=AgentStatus.success, message=state.response)
            )

        # Use exact name from DB for the update call
        state.habit_name = matches[0]["name"]
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
    """Delete a habit after the user has confirmed.

    Uses habit_id (by-ID deletion) when set — needed for disambiguating
    habits with identical names. Falls back to name-based deletion otherwise.
    """

    async def run(self, ctx: GraphRunContext[HabitGraphState, None]) -> FormatResponse:
        state = ctx.state
        if state.habit_id is not None:
            state.tool_result = await tools.delete_habit_by_id(
                state.user_id, state.habit_id
            )
        else:
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

        all_logs = await queries.get_logs(habit["id"], state.user_id, days=30)
        if not all_logs:
            state.response = f"You haven't logged '{state.habit_name}' recently — nothing to change."
            return End(AgentResponse(status=AgentStatus.success, message=state.response))

        state.existing_log_id = all_logs[0]["id"]
        state.existing_log_value = all_logs[0]["value"]

        if state.log_action == "fix":
            return AskFixLogConfirmation()

        # For delete: expose all recent logs so the user can pick by number
        state.candidate_logs = [
            {"id": r["id"], "value": r["value"], "logged_at": str(r["logged_at"])}
            for r in all_logs
        ]
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
    """Ask user to confirm deleting a log entry (not the habit itself).

    If multiple logs exist, present a numbered list so the user can pick one.
    If only one log exists, ask for direct confirmation.
    """

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        logs = state.candidate_logs or []

        if len(logs) <= 1:
            # Single log — direct confirmation
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
                        "habit_name": state.habit_name,
                    },
                )
            )

        # Multiple logs — numbered list
        lines = [f"Here are your recent **{state.habit_name}** logs. Reply with a number to select one:\n"]
        for i, log in enumerate(logs, 1):
            date_str = str(log["logged_at"])[:10]
            lines.append(f"{i}. {log['value']} — {date_str}")
        state.response = "\n".join(lines)
        return End(
            AgentResponse(
                status=AgentStatus.success,
                message=state.response,
                data={
                    "awaiting": "delete_log_select",
                    "candidate_logs": logs,
                    "habit_name": state.habit_name,
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
    """Store the format prompt for the caller to stream or run synchronously."""

    async def run(
        self, ctx: GraphRunContext[HabitGraphState, None]
    ) -> End[AgentResponse]:
        state = ctx.state
        state.format_prompt = (
            f"User said: '{state.message}'\n"
            f"Tool result: {state.tool_result}\n"
            "Write a short, friendly response."
        )
        # Return an empty placeholder — callers inspect state.format_prompt
        # and run the actual LLM call (streaming or not) themselves.
        return End(AgentResponse(status=AgentStatus.success, message=""))


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


def _habit_label(h: dict) -> str:
    """One-line description of a habit for disambiguation menus."""
    target_str = (
        f"{h['target']} {h.get('unit', '')}".strip()
        if h.get("target") is not None
        else "no target"
    )
    return f"{target_str}, {h.get('frequency', 'daily')}"


def _restore_context(state: HabitGraphState, context: dict) -> None:
    """Restore stateful fields from the client-echoed context payload."""
    state.habit_name = context.get("habit_name")
    state.habit_value = context.get("habit_value") or context.get("new_value")
    state.existing_log_id = context.get("log_id")
    state.existing_log_value = context.get("existing_value") or context.get("value")
    state.habit_id = context.get("habit_id")
    state.habit_ids = context.get("habit_ids")
    state.candidate_logs = context.get("candidate_logs")


async def _run_graph_and_get_state(
    message: str,
    user_id: int,
    conversation_id: int | None,
    awaiting: str | None,
    context: dict | None,
) -> tuple[HabitGraphState, AgentResponse]:
    """Run the graph synchronously and return (state, response) without streaming.

    Called before the SSE response starts so the endpoint can inspect
    response.data.awaiting and set HTTP response headers before the body begins.
    """
    # Load history before saving current message so it isn't included
    conversation_history: str | None = None
    if conversation_id and not awaiting:
        recent = await queries.get_recent_messages(conversation_id, user_id, limit=20)
        if recent:
            conversation_history = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in recent
            )

    state = HabitGraphState(
        message=message,
        user_id=user_id,
        awaiting=awaiting,
        conversation_history=conversation_history,
    )
    if awaiting and context:
        _restore_context(state, context)
    if conversation_id:
        await queries.add_message(conversation_id, "user", message)
    result = await habit_graph.run(start_node=ClassifyIntent(), state=state)
    return state, cast(AgentResponse, result.output)


async def stream_formatter_tokens(
    state: HabitGraphState,
    response: AgentResponse,
    conversation_id: int | None,
    user_id: int,
    message: str,
    start: float,
) -> AsyncIterator[str]:
    """Yield text tokens for the SSE body.

    If state.format_prompt is set, stream the formatter LLM token-by-token.
    Otherwise yield the direct response message as a single chunk.
    In both cases, save the final message to the DB and log the run.
    """
    if state.format_prompt:
        tokens: list[str] = []
        async with _get_formatter().run_stream(state.format_prompt) as stream:
            async for chunk in stream.stream_text(delta=True, debounce_by=None):
                if chunk:
                    tokens.append(chunk)
                    yield chunk
        response_text = "".join(tokens)
    else:
        response_text = response.message or ""
        if response_text:
            yield response_text

    if conversation_id and response_text:
        await queries.add_message(conversation_id, "assistant", response_text)
    log_agent_run(
        user_id=user_id,
        message=message,
        response=response_text,
        elapsed_ms=(perf_counter() - start) * 1000,
    )


async def run_graph_agent(
    message: str,
    user_id: int,
    conversation_id: int | None = None,
    awaiting: str | None = None,
    context: dict | None = None,
) -> AgentResponse:
    """Run the full graph and return a complete AgentResponse (non-streaming)."""
    start = perf_counter()
    state, response = await _run_graph_and_get_state(
        message, user_id, conversation_id, awaiting, context
    )

    if state.format_prompt:
        fmt = await _get_formatter().run(state.format_prompt)
        response = AgentResponse(status=AgentStatus.success, message=fmt.output)

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
    """Run the graph and stream response tokens.

    The SSE chat endpoint calls _run_graph_and_get_state + stream_formatter_tokens
    directly to control HTTP headers. This wrapper is used by the voice WebSocket
    and any other callers that need a simple async iterator interface.
    """
    start = perf_counter()
    state, response = await _run_graph_and_get_state(
        message, user_id, conversation_id, awaiting, context
    )
    async for chunk in stream_formatter_tokens(
        state, response, conversation_id, user_id, message, start
    ):
        yield chunk
