from datetime import datetime

from pydantic import BaseModel, Field


class STMMessage(BaseModel):
    message_id: str
    user_id: str
    thread_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: dict = Field(default_factory=dict)