"""Chat endpoints for AI interaction with streaming support."""

import json
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from app.agent.agent import run_agent
from app.api.deps import CurrentUser
from app.db import queries
from app.models.schemas import AgentResponse, ChatRequest

router = APIRouter(tags=["chat"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/chat", response_model=AgentResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request, req: ChatRequest, user_id: CurrentUser
) -> AgentResponse:
    """Send a message to the AI agent (non-streaming)."""
    if req.conversation_id is not None:
        conversation = await queries.get_conversation(req.conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

    return await run_agent(
        req.message, user_id, req.conversation_id, req.awaiting, req.context
    )


@router.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: ChatRequest, user_id: CurrentUser):
    """Send a message to the AI agent with streaming response (SSE).

    The graph runs synchronously before the SSE response starts so that
    `awaiting` state can be delivered as HTTP response headers — available
    to the client the moment `await fetch()` resolves, before any body bytes.
    The SSE body carries only text tokens.
    """
    from app.agent.graph_agent import _run_graph_and_get_state, stream_formatter_tokens

    if req.conversation_id is not None:
        conversation = await queries.get_conversation(req.conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

    start = perf_counter()
    state, response = await _run_graph_and_get_state(
        req.message, user_id, req.conversation_id, req.awaiting, req.context
    )

    headers: dict[str, str] = {}
    if response.data and response.data.get("awaiting"):
        headers["x-tally-awaiting"] = str(response.data["awaiting"])
        headers["x-tally-context"] = json.dumps(response.data)

    async def event_generator():
        async for chunk in stream_formatter_tokens(
            state, response, req.conversation_id, user_id, req.message, start
        ):
            yield {"event": "message", "data": chunk}
        yield {"event": "done", "data": "[DONE]"}

    # ping=0: no keep-alive pings so connection closes cleanly after [DONE].
    # sep="\n": use LF line endings — sse_starlette defaults to CRLF (\r\n)
    # which breaks the client-side "\n\n" event-boundary parser.
    sse = EventSourceResponse(event_generator(), ping=0, sep="\n")
    for k, v in headers.items():
        sse.headers[k] = v
    return sse
