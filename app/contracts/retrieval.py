from typing import Literal

from pydantic import BaseModel, Field


RetrievalSource = Literal["stm", "ltm", "pdf"]


class RetrievedItem(BaseModel):
    item_id: str
    source: RetrievalSource
    content: str
    score: float | None = None
    rank: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    source: RetrievalSource
    query: str
    items: list[RetrievedItem] = Field(default_factory=list)
    latency_ms: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    user_id: str
    thread_id: str
    query: str
    top_k: int = Field(default=5, ge=1)


class STMQuery(RetrievalRequest):
    source: Literal["stm"] = "stm"
    include_recent_messages: bool = True
    recency_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class LTMQuery(RetrievalRequest):
    source: Literal["ltm"] = "ltm"


class PDFQuery(RetrievalRequest):
    source: Literal["pdf"] = "pdf"
    document_ids: list[str] = Field(default_factory=list)


class MergedRetrievalResult(BaseModel):
    query: str
    items: list[RetrievedItem] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)