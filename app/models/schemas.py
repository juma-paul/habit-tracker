from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ===============================
# ENUMS
# ===============================


class Frequency(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class Theme(str, Enum):
    light = "light"
    dark = "dark"
    system = "system"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class AgentStatus(str, Enum):
    success = "success"
    error = "error"
    clarification = "clarification"


# ===============================
# AUTH
# ===============================


class UserResponse(BaseModel):
    id: int
    email: str
    name: str | None = None


# ===============================
# HABITS
# ===============================


class HabitBase(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    target: float | None = Field(default=None, ge=0)
    unit: str | None = Field(default=None, max_length=50)
    frequency: Frequency = Frequency.daily


class HabitCreate(HabitBase):
    pass


class HabitUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    target: float | None = Field(default=None, ge=0)
    unit: str | None = Field(default=None, max_length=50)
    frequency: Frequency | None = None


class HabitResponse(HabitBase):
    id: int
    user_id: int
    created_at: datetime


# ===============================
# LOGS
# ===============================


class LogBase(BaseModel):
    value: float = Field(ge=0)
    notes: str | None = Field(default=None, max_length=500)


class LogCreate(LogBase):
    pass


class LogUpdate(BaseModel):
    value: float | None = Field(default=None, ge=0)
    notes: str | None = Field(default=None, max_length=500)


class HabitLogResponse(LogBase):
    id: int
    habit_id: int
    logged_at: datetime


# ===============================
# CHAT / AGENT
# ===============================


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: int | None = None
    awaiting: str | None = None
    # Echo the `data` dict from the previous response back here so the server
    # can restore stateful fields (habit_name, log_id, etc.) between requests.
    context: dict[str, Any] | None = None


class AgentResponse(BaseModel):
    """Structured response from AI Agent."""

    status: AgentStatus
    message: str
    data: dict[str, Any] | None = None


# ===============================
# VOICE
# ===============================


class VoiceResponse(BaseModel):
    """Response for voice endpoints."""

    transcript: str
    agent_response: AgentResponse
    audio_url: str | None = None


# ===============================
# CONVERSATIONS
# ===============================


class ConversationBase(BaseModel):
    title: str = Field(default="New Chat", max_length=255)


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ConversationResponse(ConversationBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: MessageRole
    content: str
    created_at: datetime


class ConversationWithMessages(BaseModel):
    conversation: ConversationResponse
    messages: list[MessageResponse]


# ===============================
# USER SETTINGS
# ===============================


class SettingsUpdate(BaseModel):
    theme: Theme | None = None
    voice_enabled: bool | None = None
    notifications: bool | None = None


class UserSettingsResponse(BaseModel):
    id: int
    user_id: int
    theme: Theme
    voice_enabled: bool
    notifications: bool
