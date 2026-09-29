from __future__ import annotations

from uuid import uuid4

from app.contracts.runtime import AgentResponse


class Phase0Graph:
    """Minimal graph boundary used until the real LangGraph is implemented."""

    def invoke(self, *, user_id: str, thread_id: str, query: str) -> AgentResponse:
        return AgentResponse(
            user_id=user_id,
            thread_id=thread_id,
            answer=(
                "Phase 0 skeleton is running. Retrieval and the full LangGraph "
                "workflow will be implemented in later phases."
            ),
            trace_id=f"trace_{uuid4().hex}",
        )
