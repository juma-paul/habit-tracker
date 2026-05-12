"""Agent entry points — all traffic routes to the pydantic_graph agent."""

from collections.abc import AsyncIterator

from app.models.schemas import AgentResponse


async def run_agent(
    message: str,
    user_id: int,
    conversation_id: int | None = None,
    awaiting: str | None = None,
    context: dict | None = None,
) -> AgentResponse:
    """Run the graph agent and return a complete AgentResponse."""
    from app.agent.graph_agent import (
        run_graph_agent,  # lazy import avoids circular deps
    )

    return await run_graph_agent(message, user_id, conversation_id, awaiting, context)


async def run_agent_stream(
    message: str,
    user_id: int,
    conversation_id: int | None = None,
    awaiting: str | None = None,
    context: dict | None = None,
) -> AsyncIterator[str]:
    """Run the graph agent and stream response tokens."""
    from app.agent.graph_agent import (
        run_graph_stream,  # lazy import avoids circular deps
    )

    async for chunk in run_graph_stream(
        message, user_id, conversation_id, awaiting, context
    ):
        yield chunk
