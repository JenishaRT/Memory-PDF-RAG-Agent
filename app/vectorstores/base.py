from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class VectorItem(BaseModel):
    id: str
    content: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorQuery(BaseModel):
    query_vector: list[float]
    top_k: int = Field(ge=1)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)


class VectorMatch(BaseModel):
    id: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStore(Protocol):
    def add(self, items: list[VectorItem]) -> None:
        ...

    def search(self, query: VectorQuery) -> list[VectorMatch]:
        ...

    def update(self, items: list[VectorItem]) -> None:
        ...

    def delete(self, ids: list[str]) -> None:
        ...
