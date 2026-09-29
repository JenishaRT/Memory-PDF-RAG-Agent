from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from app.contracts.conversation import ConversationMessage
from app.contracts.runtime import AgentResponse
from app.graph.graph import Phase0Graph
from app.runtime.conversation_store import ConversationStore


class AgentRuntimeProtocol(Protocol):
    def run(self, user_id: str, thread_id: str, message: str) -> AgentResponse:
        ...


class AgentRuntime:
    def __init__(
        self,
        graph: Phase0Graph,
        conversation_store: ConversationStore,
    ) -> None:
        self.graph = graph
        self.conversation_store = conversation_store

    def run(self, user_id: str, thread_id: str, message: str) -> AgentResponse:
        self.conversation_store.append(
            ConversationMessage(
                message_id=f"msg_{uuid4().hex}",
                role="user",
                content=message,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                thread_id=thread_id,
            )
        )

        response = self.graph.invoke(
            user_id=user_id,
            thread_id=thread_id,
            query=message,
        )

        self.conversation_store.append(
            ConversationMessage(
                message_id=f"msg_{uuid4().hex}",
                role="assistant",
                content=response.answer,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                thread_id=thread_id,
            )
        )

        return response
