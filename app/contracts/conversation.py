from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


MessageRole = Literal["system", "user", "assistant", "tool"]


class ConversationMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid4()))

    user_id: str
    thread_id: str
    role: MessageRole
    content: str

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, object] = Field(default_factory=dict)


class Conversation(BaseModel):
    user_id: str
    thread_id: str

    messages: list[ConversationMessage] = Field(default_factory=list)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, object] = Field(default_factory=dict)