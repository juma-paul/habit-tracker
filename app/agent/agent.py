"""PydanticAI agent configuration with logging and observability."""

from functools import lru_cache
from time import perf_counter
from typing import AsyncIterator

from loguru import logger
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from app.core.config import get_settings
from app.core.logging import log_tool_call, log_agent_run
from app.models.schemas import AgentResponse
from app.agent.prompt import SYSTEM_PROMPT
from app.agent import tools
from app.db import queries


@lru_cache
def get_agent() -> Agent[int, AgentResponse]:
    """Get or create the agent (lazy initialization)."""
    settings = get_settings()

    model: AnthropicModel | OpenAIChatModel
    if settings.ai_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when AI_PROVIDER=anthropic")
        model = AnthropicModel(
            settings.anthropic_model,
            provider=AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value()
            ),
        )
    else:
        model = OpenAIChatModel(
            settings.openai_model,
            provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value()),
        )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        deps_type=int,
        output_type=AgentResponse,
    )

    @agent.tool
    async def create_habit(
        ctx: RunContext[int],
        name: str,
        target: float | None = None,
        unit: str | None = None,
        frequency: str = "daily",
    ) -> dict:
        """Create a new habit to track.

        Args:
            name: Name of the habit (e.g., "running", "reading").
            target: Optional target value (e.g., 5 for 5 km).
            unit: Optional unit of measurement (e.g., "km", "minutes").
            frequency: How often to track — "daily", "weekly", or "monthly".
        """
        return await log_tool_call(tools.create_habit)(
            ctx.deps, name, target, unit, frequency
        )

    @agent.tool
    async def list_habits(ctx: RunContext[int]) -> dict:
        """List all active habits for the user."""
        return await log_tool_call(tools.list_habits)(ctx.deps)

    @agent.tool
    async def log_activity(
        ctx: RunContext[int], habit_name: str, value: float, notes: str | None = None
    ) -> dict:
        """Log activity for a habit.

        Args:
            habit_name: Name of the habit to log for (partial match supported).
            value: The value to log (e.g., 3 for a 3 km run).
            notes: Optional notes about this activity.
        """
        return await log_tool_call(tools.log_activity)(
            ctx.deps, habit_name, value, notes
        )

    @agent.tool
    async def get_progress(
        ctx: RunContext[int], habit_name: str, days: int = 7
    ) -> dict:
        """Get progress stats for a habit over a time period.

        Args:
            habit_name: Name of the habit to check (partial match supported).
            days: Number of days to look back (default: 7).
        """
        return await log_tool_call(tools.get_progress)(ctx.deps, habit_name, days)

    @agent.tool
    async def update_habit(
        ctx: RunContext[int],
        habit_name: str,
        new_name: str | None = None,
        target: float | None = None,
        unit: str | None = None,
        frequency: str | None = None,
    ) -> dict:
        """Update a habit's settings.

        Args:
            habit_name: Current name of the habit to update.
            new_name: New name for the habit (optional).
            target: New target value (optional).
            unit: New unit of measurement (optional).
            frequency: New frequency — "daily", "weekly", or "monthly" (optional).
        """
        return await log_tool_call(tools.update_habit)(
            ctx.deps, habit_name, new_name, target, unit, frequency
        )

    @agent.tool
    async def delete_habit(ctx: RunContext[int], habit_name: str) -> dict:
        """Delete a habit permanently.

        Args:
            habit_name: Name of the habit to delete (partial match supported).
        """
        return await log_tool_call(tools.delete_habit)(ctx.deps, habit_name)

    @agent.tool
    async def update_log(
        ctx: RunContext[int],
        log_id: int,
        value: float | None = None,
        notes: str | None = None,
    ) -> dict:
        """Update a previous log entry.

        Args:
            log_id: ID of the log entry to update.
            value: New value (optional).
            notes: New notes (optional).
        """
        return await log_tool_call(tools.update_log)(ctx.deps, log_id, value, notes)

    @agent.tool
    async def delete_log(ctx: RunContext[int], log_id: int) -> dict:
        """Delete a log entry.

        Args:
            log_id: ID of the log entry to delete.
        """
        return await log_tool_call(tools.delete_log)(ctx.deps, log_id)

    model_name = (
        settings.anthropic_model
        if settings.ai_provider == "anthropic"
        else settings.openai_model
    )
    logger.info(f"Agent initialized with model: {model_name}")
    return agent


async def _run_freeform(
    message: str, user_id: int, conversation_id: int | None = None
) -> AgentResponse:
    """Freeform agent: single LLM call, model decides tool order.

    If conversation_id is provided, saves the exchange to the database.
    """
    start = perf_counter()
    agent = get_agent()

    logger.debug(f"Running agent for user {user_id}: {message[:50]}...")

    result = await agent.run(
        message,
        deps=user_id,
        usage_limits=UsageLimits(request_limit=15, tool_calls_limit=5),
    )
    elapsed_ms = (perf_counter() - start) * 1000

    usage = result.usage()
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0

    log_agent_run(
        user_id=user_id,
        message=message,
        response=result.output.message,
        elapsed_ms=elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    if conversation_id:
        await queries.add_message(conversation_id, "user", message)
        await queries.add_message(conversation_id, "assistant", result.output.message)

    return result.output


async def _run_freeform_stream(
    message: str, user_id: int, conversation_id: int | None = None
) -> AsyncIterator[str]:
    """Freeform streaming: single API call, model decides tool order.

    Yields text chunks as they arrive via stream_text(delta=True).
    """
    start = perf_counter()
    agent = get_agent()

    logger.debug(f"Starting stream for user {user_id}: {message[:50]}...")

    history: list[ModelRequest | ModelResponse] = []
    if conversation_id:
        recent = await queries.get_recent_messages(conversation_id, user_id, limit=10)
        for msg in recent:
            if msg["role"] == "user":
                history.append(
                    ModelRequest(parts=[UserPromptPart(content=msg["content"])])
                )
            else:
                history.append(ModelResponse(parts=[TextPart(content=msg["content"])]))

    if conversation_id:
        await queries.add_message(conversation_id, "user", message)

    full_response: list[str] = []

    async with agent.run_stream(
        message,
        deps=user_id,
        output_type=str,
        message_history=history,
        usage_limits=UsageLimits(request_limit=15, tool_calls_limit=5),
    ) as result:
        async for chunk in result.stream_text(delta=True):
            if chunk:
                full_response.append(chunk)
                yield chunk

    elapsed_ms = (perf_counter() - start) * 1000
    response_text = "".join(full_response)

    if conversation_id and response_text:
        await queries.add_message(conversation_id, "assistant", response_text)

    log_agent_run(
        user_id=user_id,
        message=message,
        response=response_text,
        elapsed_ms=elapsed_ms,
    )


async def run_agent(
    message: str,
    user_id: int,
    conversation_id: int | None = None,
    awaiting: str | None = None,
    context: dict | None = None,
) -> AgentResponse:
    """Route to graph or freeform agent based on CONTROL_MODEL setting."""
    from app.agent.graph_agent import run_graph_agent  # lazy import avoids circular deps

    if get_settings().control_model == "graph":
        return await run_graph_agent(message, user_id, conversation_id, awaiting, context)
    return await _run_freeform(message, user_id, conversation_id)


async def run_agent_stream(
    message: str,
    user_id: int,
    conversation_id: int | None = None,
    awaiting: str | None = None,
    context: dict | None = None,
) -> AsyncIterator[str]:
    """Route streaming to graph or freeform agent based on CONTROL_MODEL setting."""
    from app.agent.graph_agent import run_graph_stream  # lazy import avoids circular deps

    if get_settings().control_model == "graph":
        async for chunk in run_graph_stream(message, user_id, conversation_id, awaiting, context):
            yield chunk
    else:
        async for chunk in _run_freeform_stream(message, user_id, conversation_id):
            yield chunk
