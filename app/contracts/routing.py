from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    use_stm: bool = False
    use_ltm: bool = False
    use_pdf: bool = False
    query_type: str | None = None
    entities: list[str] = Field(default_factory=list)
    document_references: list[str] = Field(default_factory=list)
    memory_references: list[str] = Field(default_factory=list)
    conversation_references: list[str] = Field(default_factory=list)
