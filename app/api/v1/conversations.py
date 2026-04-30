"""Conversation endpoints for chat history management."""

from fastapi import APIRouter, HTTPException

from app.db import queries
from app.models.schemas import (
    ConversationResponse,
    ConversationCreate,
    ConversationUpdate,
    ConversationWithMessages,
    MessageResponse,
)
from app.api.deps import CurrentUser


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(user_id: CurrentUser) -> list[ConversationResponse]:
    """Get all conversations for the current user."""
    conversations = await queries.get_conversations(user_id)
    return [ConversationResponse(**c) for c in conversations]


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    user_id: CurrentUser, req: ConversationCreate | None = None
) -> ConversationResponse:
    """Create a new conversation."""
    title = req.title if req else "New Chat"
    conversation = await queries.create_conversation(user_id, title)
    return ConversationResponse(**conversation)


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: int, user_id: CurrentUser
) -> ConversationWithMessages:
    """Get a conversation with all it's messages."""
    conversation = await queries.get_conversation(conversation_id, user_id)
    if not conversation:
        raise HTTPException(404, "Conversation not found")

    messages = await queries.get_messages(conversation_id, user_id)

    return ConversationWithMessages(
        conversation=ConversationResponse(**conversation),
        messages=[MessageResponse(**m) for m in messages],
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int, req: ConversationUpdate, user_id: CurrentUser
) -> ConversationResponse:
    """Update a conversation (e.g., rename it)."""
    conversation = await queries.update_conversation(
        conversation_id, user_id, title=req.title
    )
    if not conversation:
        raise HTTPException(404, "Conversation not found")
    return ConversationResponse(**conversation)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: int, user_id: CurrentUser):
    """Delete a conversation and all it's messages."""
    result = await queries.delete_conversation(conversation_id, user_id)
    if not result:
        raise HTTPException(404, "Conversation not found")
