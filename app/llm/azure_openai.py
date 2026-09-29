from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from app.contracts.errors import ConfigurationError
from app.llm.provider import ChatMessage, LLMResponse

T = TypeVar("T", bound=BaseModel)


class AzureOpenAIProvider:
    """Phase 0 placeholder for the Azure OpenAI implementation.

    The provider contract is intentionally defined before the concrete
    integration is implemented.
    """

    def __init__(self, **_: Any) -> None:
        raise ConfigurationError(
            "AzureOpenAIProvider is not implemented until a later phase."
        )

    def invoke(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def structured_output(
        self,
        messages: list[ChatMessage],
        output_schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        raise NotImplementedError
