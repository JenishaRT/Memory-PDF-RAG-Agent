from __future__ import annotations

from pydantic import BaseModel


class AgentRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class AgentResponse(BaseModel):
    user_id: str
    thread_id: str
    answer: str
    trace_id: str
