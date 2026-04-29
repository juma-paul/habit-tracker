"""Chat endpoints for AI interaction with streaming support."""

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from app.agent.agent import run_agent, run_agent_stream
from app.agent.help import HELP_CONTENT, HELP_DATA
from app.models.schemas import ChatRequest, AgentResponse, AgentStatus
from app.api.deps import CurrentUser
from app.db import queries


router = APIRouter(tags=["chat"])
limiter = Limiter(key_func=get_remote_address)


def _is_help_command(message: str) -> bool:
    """Check if message is a help command."""
    msg = message.strip().lower()
    return msg in ["/help", "/h", "?"]


def _get_help_response() -> AgentResponse:
    """Return help content as AgentResponse."""
    return AgentResponse(
        status=AgentStatus.success,
        message=HELP_CONTENT,
        data={"type": "help", "help": HELP_DATA}
    )


@router.post("/chat", response_model=AgentResponse)
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest, user_id: CurrentUser) -> AgentResponse:
    """
    Send a message to the AI agent (non-streaming).

    Returns complete response after processing
    Use /help to see available commands
    """
    # Handle help command
    if _is_help_command(req.message):
        return _get_help_response()

    if req.conversation_id is not None:
        conversation = await queries.get_conversation(req.conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

    return await run_agent(req.message, user_id, req.conversation_id)


@router.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: ChatRequest, user_id: CurrentUser):
    """
    Send a message to the AI agent with streaming response (SSE).

    Returns Server-Sent Events with text chunks as they're generated.

    Example usage with Javascript:
    ```javascript
    const eventSource = new EventSource('/api/v1/chat/stream', {
        method: 'POST',
        body: JSON.stringify({ message: "I walked 9880 steps" })
    })

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.text); // Append to UI
    };
    ```
    or with fetch: 
    ```javascript
    const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer <token>'
        },
        body: JSON.stringify({ message: 'I walked 5000 steps' })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        console.log(decoder.decode(value));
    }
    ```
    """
    # Handle help command - return immediately without streaming
    # Help is NOT saved to conversation - it's static content
    if _is_help_command(req.message):
        async def help_generator():
            yield {"event": "message", "data": HELP_CONTENT}
            yield {"event": "done", "data": "[DONE]"}
        return EventSourceResponse(help_generator())

    if req.conversation_id is not None:
        conversation = await queries.get_conversation(req.conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

    async def event_generator():
        """Generate SSE events from agent stream."""
        try:
            async for chunk in run_agent_stream(req.message, user_id, req.conversation_id):
                yield {
                    "event": "message",
                    "data": chunk
                }
                yield {
                    "event": "done",
                    "data": "[DONE]"
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": str(e)
            }

    return EventSourceResponse(event_generator())
    