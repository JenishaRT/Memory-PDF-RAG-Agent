from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str | None = None
    usage: dict[str, Any] = {}


class LLMProvider(Protocol):
    def invoke(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        ...

    def structured_output(
        self,
        messages: list[ChatMessage],
        output_schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        ...
