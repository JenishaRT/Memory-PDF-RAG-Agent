from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    PLAN = "plan"
    TECHNICAL_STACK = "technical_stack"
    FACT = "fact"
    CONSTRAINT = "constraint"
    WORKFLOW = "workflow"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INACTIVE = "inactive"


class MemorySource(BaseModel):
    thread_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    memory_id: str
    user_id: str
    memory_type: MemoryType
    content: str
    importance: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = Field(default=1, ge=1)
    source: MemorySource = Field(default_factory=MemorySource)


class CandidateMemory(BaseModel):
    candidate_id: str
    user_id: str
    memory_type: MemoryType
    content: str
    importance: float = Field(ge=0.0, le=1.0)
    source: MemorySource = Field(default_factory=MemorySource)


class MemoryAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    IGNORE = "ignore"


class MemoryDecision(BaseModel):
    action: MemoryAction
    candidate_id: str
    related_memory_ids: list[str] = Field(default_factory=list)
    reason_code: str
