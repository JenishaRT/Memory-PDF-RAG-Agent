from __future__ import annotations

from pydantic import BaseModel, Field

from app.contracts.retrieval import RetrievedItem


class Context(BaseModel):
    stm: list[RetrievedItem] = Field(default_factory=list)
    ltm: list[RetrievedItem] = Field(default_factory=list)
    documents: list[RetrievedItem] = Field(default_factory=list)
    token_budget: int = Field(ge=1)
    estimated_tokens: int = Field(default=0, ge=0)
