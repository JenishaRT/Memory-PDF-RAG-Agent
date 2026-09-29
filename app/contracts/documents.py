from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str
    filename: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None = None
    section: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
