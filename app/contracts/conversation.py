from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    message_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    user_id: str
    thread_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    user_id: str
    thread_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
