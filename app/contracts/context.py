from typing import Literal

from pydantic import BaseModel, Field


ContextSource = Literal["stm", "ltm", "pdf"]


class ContextItem(BaseModel):
    source: ContextSource
    content: str
    item_id: str
    score: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class AgentContext(BaseModel):
    items: list[ContextItem] = Field(default_factory=list)
    max_tokens: int = 6000
    estimated_tokens: int = 0

    def add(self, item: ContextItem) -> None:
        self.items.append(item)

    @property
    def is_empty(self) -> bool:
        return not self.items