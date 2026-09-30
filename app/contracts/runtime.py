from typing import Protocol

from pydantic import BaseModel, Field

from app.contracts.conversation import Conversation, ConversationMessage


class AgentRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    user_id: str
    thread_id: str
    answer: str
    trace_id: str
    metadata: dict[str, object] = Field(default_factory=dict)


class ConversationStore(Protocol):
    def load(self, user_id: str, thread_id: str) -> Conversation:
        ...

    def append_message(self, message: ConversationMessage) -> None:
        ...


class GraphRunner(Protocol):
    def run(
        self,
        request: AgentRequest,
        trace_id: str,
    ) -> AgentResponse:
        ...