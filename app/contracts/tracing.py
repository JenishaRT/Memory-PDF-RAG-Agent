from datetime import datetime, timezone

from pydantic import BaseModel, Field


class RetrievalTrace(BaseModel):
    source: str
    query: str

    retrieved_ids: list[str] = Field(default_factory=list)
    scores: list[float | None] = Field(default_factory=list)
    ranks: list[int | None] = Field(default_factory=list)

    latency_ms: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class MemoryTrace(BaseModel):
    operation: str
    user_id: str
    thread_id: str | None = None

    memory_ids: list[str] = Field(default_factory=list)
    action: str | None = None

    latency_ms: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class GraphTrace(BaseModel):
    trace_id: str
    user_id: str
    thread_id: str
    query: str

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None

    planner: dict[str, object] = Field(default_factory=dict)

    stm: RetrievalTrace | None = None
    ltm: RetrievalTrace | None = None
    pdf_rag: RetrievalTrace | None = None

    reranker: dict[str, object] = Field(default_factory=dict)
    context: dict[str, object] = Field(default_factory=dict)
    answer: dict[str, object] = Field(default_factory=dict)
    validation: dict[str, object] = Field(default_factory=dict)

    metadata: dict[str, object] = Field(default_factory=dict)












# from __future__ import annotations

# from datetime import datetime
# from typing import Any

# from pydantic import BaseModel, Field

# from app.contracts.routing import RetrievalPlan
# from app.contracts.runtime import AgentResponse


# class TraceBase(BaseModel):
#     trace_id: str
#     user_id: str
#     thread_id: str
#     timestamp: datetime
#     latency_ms: float = 0.0


# class STMTrace(TraceBase):
#     query: str
#     top_k: int
#     retrieved_message_ids: list[str] = Field(default_factory=list)
#     scores: list[float | None] = Field(default_factory=list)
#     ranks: list[int] = Field(default_factory=list)
#     recency_applied: bool = False
#     context_expansion_applied: bool = False


# class LTMTrace(TraceBase):
#     query: str
#     retrieved_memory_ids: list[str] = Field(default_factory=list)
#     scores: list[float | None] = Field(default_factory=list)
#     ranks: list[int] = Field(default_factory=list)
#     memory_statuses: list[str] = Field(default_factory=list)
#     consolidation_action: str | None = None


# class PDFTrace(TraceBase):
#     query: str
#     document_ids: list[str] = Field(default_factory=list)
#     filenames: list[str] = Field(default_factory=list)
#     page_numbers: list[int | None] = Field(default_factory=list)
#     chunk_ids: list[str] = Field(default_factory=list)
#     similarity_scores: list[float | None] = Field(default_factory=list)
#     ranks: list[int] = Field(default_factory=list)
#     reranker_scores: list[float | None] = Field(default_factory=list)


# class ValidationResult(BaseModel):
#     valid: bool
#     needs_retry: bool
#     reason: str


# class GraphTrace(BaseModel):
#     trace_id: str
#     user_id: str
#     thread_id: str
#     query: str
#     planner: RetrievalPlan | None = None
#     stm_trace_id: str | None = None
#     ltm_trace_id: str | None = None
#     pdf_trace_id: str | None = None
#     reranker: dict[str, Any] | None = None
#     context: dict[str, Any] | None = None
#     answer: dict[str, Any] | None = None
#     validation: ValidationResult | None = None
