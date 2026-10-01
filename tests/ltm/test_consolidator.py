import json

import pytest

from app.contracts.memory import (
    CandidateMemory,
    MemoryRecord,
    MemorySource,
)
from app.llm.provider import ChatMessage, LLMResponse
from app.memory.ltm.consolidator import LTMConsolidator


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None
        self.output_schema = None

    def structured_output(
        self,
        messages: list[ChatMessage],
        output_schema,
        *,
        temperature: float | None = None,
    ):
        self.messages = messages
        self.output_schema = output_schema
        return output_schema.model_validate(json.loads(self.response))


def _candidate(
    *,
    content: str = "User prefers Qdrant.",
) -> CandidateMemory:
    return CandidateMemory(
        user_id="user_001",
        memory_type="preference",
        content=content,
        confidence=0.95,
        source=MemorySource(
            thread_id="thread_002",
            message_ids=["message_002"],
        ),
    )


def _memory(
    *,
    memory_id: str = "memory_001",
    content: str = "User prefers Chroma.",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="user_001",
        memory_type="preference",
        content=content,
        source=MemorySource(
            thread_id="thread_001",
            message_ids=["message_001"],
        ),
    )


def _response(
    *,
    action: str,
    memory_ids: list[str],
    reason: str = "Test decision.",
) -> str:
    return json.dumps(
        {
            "action": action,
            "existing_memory_ids": memory_ids,
            "reason": reason,
        }
    )


def test_no_related_memory_returns_add() -> None:
    llm = FakeLLM(
        _response(
            action="ADD",
            memory_ids=[],
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    candidate = _candidate()

    decision = consolidator.consolidate(
        candidate=candidate,
        related_memories=[],
    )

    assert decision.action == "ADD"
    assert decision.candidate == candidate
    assert decision.existing_memory_ids == []
    assert llm.messages is None


def test_duplicate_can_be_ignored() -> None:
    memory = _memory(
        content="User prefers Qdrant."
    )

    llm = FakeLLM(
        _response(
            action="IGNORE",
            memory_ids=[memory.memory_id],
            reason="The candidate duplicates the existing memory.",
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    decision = consolidator.consolidate(
        candidate=_candidate(
            content="User prefers Qdrant."
        ),
        related_memories=[memory],
    )

    assert decision.action == "IGNORE"
    assert llm.output_schema.__name__ == "LTMConsolidationOutput"
    assert decision.existing_memory_ids == [
        "memory_001"
    ]


def test_updated_memory_can_produce_update() -> None:
    memory = _memory(
        content="User is learning Python."
    )

    llm = FakeLLM(
        _response(
            action="UPDATE",
            memory_ids=[memory.memory_id],
            reason="The candidate provides a newer version.",
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    decision = consolidator.consolidate(
        candidate=_candidate(
            content="User is now working professionally with Python."
        ),
        related_memories=[memory],
    )

    assert decision.action == "UPDATE"
    assert decision.existing_memory_ids == [
        "memory_001"
    ]


def test_complementary_memory_can_produce_merge() -> None:
    memory = _memory(
        content="User prefers Python."
    )

    llm = FakeLLM(
        _response(
            action="MERGE",
            memory_ids=[memory.memory_id],
            reason="The two memories describe complementary information.",
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    decision = consolidator.consolidate(
        candidate=_candidate(
            content="User uses Python for backend development."
        ),
        related_memories=[memory],
    )

    assert decision.action == "MERGE"


def test_contradictory_memory_can_produce_supersede() -> None:
    memory = _memory(
        content="User prefers Chroma."
    )

    llm = FakeLLM(
        _response(
            action="SUPERSEDE",
            memory_ids=[memory.memory_id],
            reason="The candidate indicates a changed preference.",
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    decision = consolidator.consolidate(
        candidate=_candidate(
            content="User now prefers Qdrant."
        ),
        related_memories=[memory],
    )

    assert decision.action == "SUPERSEDE"
    assert decision.existing_memory_ids == [
        "memory_001"
    ]


def test_add_can_be_returned_when_related_memory_is_not_relevant() -> None:
    memory = _memory(
        content="User's favorite programming language is Python."
    )

    llm = FakeLLM(
        _response(
            action="ADD",
            memory_ids=[],
            reason="The candidate represents a different fact.",
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    decision = consolidator.consolidate(
        candidate=_candidate(
            content="User prefers Qdrant."
        ),
        related_memories=[memory],
    )

    assert decision.action == "ADD"
    assert decision.existing_memory_ids == []


def test_consolidator_sends_candidate_and_existing_memories() -> None:
    memory = _memory()

    llm = FakeLLM(
        _response(
            action="IGNORE",
            memory_ids=[memory.memory_id],
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    consolidator.consolidate(
        candidate=_candidate(),
        related_memories=[memory],
    )

    prompt = llm.messages[1].content

    assert "User prefers Qdrant." in prompt
    assert "User prefers Chroma." in prompt
    assert "memory_001" in prompt


def test_consolidator_rejects_invalid_action() -> None:
    llm = FakeLLM(
        _response(
            action="DELETE",
            memory_ids=["memory_001"],
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    with pytest.raises(
        ValueError,
    ):
        consolidator.consolidate(
            candidate=_candidate(),
            related_memories=[_memory()],
        )


def test_consolidator_rejects_unknown_memory_id() -> None:
    llm = FakeLLM(
        _response(
            action="SUPERSEDE",
            memory_ids=["unknown_memory"],
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    with pytest.raises(
        ValueError,
        match="unknown memory ID",
    ):
        consolidator.consolidate(
            candidate=_candidate(),
            related_memories=[_memory()],
        )


def test_non_add_action_requires_existing_memory() -> None:
    llm = FakeLLM(
        _response(
            action="UPDATE",
            memory_ids=[],
        )
    )

    consolidator = LTMConsolidator(llm=llm)

    with pytest.raises(
        ValueError,
        match="existing_memory_ids",
    ):
        consolidator.consolidate(
            candidate=_candidate(),
            related_memories=[_memory()],
        )