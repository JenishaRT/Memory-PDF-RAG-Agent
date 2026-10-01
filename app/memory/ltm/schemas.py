from pydantic import BaseModel, Field

from app.contracts.memory import MemoryAction, MemoryType


class ExtractedMemory(BaseModel):
    memory_type: MemoryType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    message_ids: list[str]


class LTMExtractionOutput(BaseModel):
    memories: list[ExtractedMemory]


class LTMConsolidationOutput(BaseModel):
    action: MemoryAction
    existing_memory_ids: list[str] = Field(default_factory=list)
    reason: str | None = None