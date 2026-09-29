from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["stm", "ltm", "pdf"]


class RetrievedItem(BaseModel):
    source: SourceType
    id: str
    content: str
    score: float | None = None
    rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalFailure(BaseModel):
    error_type: str
    message: str
    recoverable: bool = True


class RetrievalResult(BaseModel):
    source: SourceType
    query: str
    items: list[RetrievedItem] = Field(default_factory=list)
    result_count: int = 0
    latency_ms: float = 0.0
    trace_id: str
    error: RetrievalFailure | None = None


class STMQuery(BaseModel):
    user_id: str
    thread_id: str
    query: str
    top_k: int = Field(ge=1)
    recency_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    score_threshold: float | None = None


class LTMQuery(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(ge=1)
    score_threshold: float | None = None


class PDFQuery(BaseModel):
    query: str
    user_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(ge=1)
    score_threshold: float | None = None


class STMResult(RetrievalResult):
    source: Literal["stm"] = "stm"


class LTMResult(RetrievalResult):
    source: Literal["ltm"] = "ltm"


class RAGResult(RetrievalResult):
    source: Literal["pdf"] = "pdf"
