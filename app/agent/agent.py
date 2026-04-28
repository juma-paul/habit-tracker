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
    PartDeltaEvent,
    TextPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
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

    # Build model based on configured provider
    if settings.ai_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when AI_PROVIDER=anthropic")
        model = AnthropicModel(
            settings.anthropic_model,
            provider=AnthropicProvider(api_key=settings.anthropic_api_key.get_secret_value()),
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
        output_type=AgentResponse
    )

    # Register tools with logging decorators
    @agent.tool
    async def create_habit(
        ctx: RunContext[int],
        name: str,
        target: float | None = None,
        unit: str | None = None,
        frequency: str = "daily"
    ) -> dict:
        """Create a new habit to track.
        
        Args:
            ctx: Agent context with user_id
            name: Name of the habit (e.g., "running", "reading")
            target: Optional target value (e.g., 5 for 5km)
            unit: Optional unit of measurement (e.g., "km", "minutes")
            frequency: How often to track - "daily", "weekly", or "monthly"
        """
        return await log_tool_call(tools.create_habit)(ctx.deps, name, target, unit, frequency)
    

    @agent.tool
    async def list_habits(ctx: RunContext[int]) -> dict:
        """List all active habits for the user.
        
        Args:
            ctx: Agent context with user_id
        """
        return await log_tool_call(tools.list_habits)(ctx.deps)
    

    @agent.tool
    async def log_activity(
        ctx: RunContext[int],
        habit_name: str,
        value: float,
        notes: str | None = None
    ) -> dict:
        """Log activity for a habit.
        
        Args:
            ctx: Agent context with user_id
            habit_name: Name of the habit to log for (partial match supported)
            value: The value to log (e.g., 3 for 3km run)
            notes: Optional notes about this activity
        """
        return await log_tool_call(tools.log_activity)(ctx.deps, habit_name, value, notes)
    
    
    @agent.tool
    async def get_progress(
        ctx: RunContext[int],
        habit_name: str,
        days: int = 7
    ) -> dict:
        """Get progress stats for a habit over a time period.
        
        Args:
            ctx: Agent context with user_id
            habit_name: name of the habit to check (partial match supported)
            days: Number of days to look back (default: 7)
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
            ctx: Agent context with user_id
            habit_name: Current name of the habit to update
            new_name: New name for the habit(optional)
            target: New target vakue (optional)
            unit: New unit of measurement (optional)
            frequency: New frequency - "daily", "weekly", or "monthly" (optional)
        """
        return await log_tool_call(tools.update_habit)(
            ctx.deps, habit_name, new_name, target, unit, frequency
        )
    

    @agent.tool
    async def delete_habit(ctx: RunContext[int], habit_name: str) -> dict:
        """Delete a habit permanently.
        
        Args:
            ctx: Agent context with user_id
            habit_name: Name of the habit to delete (partial match supported)
        """
        return await log_tool_call(tools.delete_habit)(ctx.deps, habit_name)
    

    @agent.tool
    async def update_log(
        ctx: RunContext[int],
        log_id: int,
        value: float | None = None,
        notes: str | None = None
    ) -> dict:
        """Update a previous log entry.
        
        Args:
            ctx: Agent context with user_id
            log_id: ID of the log entry to update
            value: New value (optional)
            notes: New notes (optional)
        """
        return await log_tool_call(tools.update_log)(ctx.deps, log_id, value, notes)
    

    @agent.tool
    async def delete_log(ctx: RunContext[int], log_id: int) -> dict:
        """Delete a log entry.
        
        Args:
            ctx: Agent context with user_id
            log_id: ID of the log entry to delete
        """
        return await log_tool_call(tools.delete_log)(ctx.deps, log_id)
    
    model_name = settings.anthropic_model if settings.ai_provider == "anthropic" else settings.openai_model
    logger.info(f"Agent initialized with model: {model_name}")
    return agent



async def run_agent(
        message: str,
        user_id: int,
        conversation_id: int | None = None
) -> AgentResponse:
    """
    Run the agent with a user message.

    If conversation_id is provided, saves the exchange to the database.
    """
    start = perf_counter()
    agent = get_agent()

    logger.debug(f"Running agent for user {user_id}: {message[:50]}...")

    result = await agent.run(message, deps=user_id, usage_limits=UsageLimits(request_limit=15, tool_calls_limit=5))
    elapsed_ms = (perf_counter() - start) * 1000

    # Extract usage info if available
    usage = result.usage()
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0

    # Log the run with metrics
    log_agent_run(
        user_id=user_id,
        message=message,
        response=result.output.message,
        elapsed_ms=elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    # Save messages to conversation if provided
    if conversation_id:
        await queries.add_message(conversation_id, "user", message)
        await queries.add_message(conversation_id, "assistant", result.output.message)
    
    return result.output


async def run_agent_stream(
        message: str,
        user_id: int,
        conversation_id: int | None = None
) -> AsyncIterator[str]:
    """
    Stream agent response using PydanticAI's native streaming.
    Single API call. Tools execute via PydanticAI's internal loop
    Yields text chunks as they arrive.
    """
    start = perf_counter()
    agent = get_agent()

    logger.debug(f"Starting stream for user {user_id}: {message[:50]}...")

    # Load conversation history if available
    history: list[ModelRequest | ModelResponse] = []
    if conversation_id:
        recent = await queries.get_recent_messages(conversation_id, user_id, limit=10)
        for msg in recent:
            if msg["role"] == "user":
                history.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
            else:
                history.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    
    # Save user message before streaming starts
    if conversation_id:
        await queries.add_message(conversation_id, "user", message)
    
    full_response: list[str] = []
    tool_call_count = 0

    async with agent.run_stream(
        message,
        deps=user_id,
        output_type=str,
        message_history=history,
        usage_limits=UsageLimits(request_limit=15, tool_calls_limit=5),
    ) as result:
        async for event in result.stream_events():
            if isinstance(event, FunctionToolCallEvent):
                tool_call_count += 1
                logger.info(
                    f"Stream tool call: {event.part.tool_name}",
                    tool=event.part.tool_name,
                    args=event.part.args_as_dict(),
                )
            elif isinstance(event, FunctionToolResultEvent):
                logger.info(
                    f"Stream tool result: {event.result.tool_name}",
                    tool=event.result.tool_name,
                    content=str(event.result.content)[:200],
                )
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                chunk = event.delta.content_delta
                if chunk:
                    full_response.append(chunk)
                    yield chunk

    # After stream complete - save and log
    elapsed_ms = (perf_counter() - start) * 1000
    response_text = "".join(full_response)

    if conversation_id and response_text:
        await queries.add_message(conversation_id, "assistant", response_text)

    log_agent_run(
        user_id=user_id,
        message=message,
        response=response_text,
        elapsed_ms=elapsed_ms,
        tool_calls=tool_call_count
    )
    