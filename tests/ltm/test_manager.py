from app.contracts.memory import (
    CandidateMemory,
    MemoryDecision,
    MemoryRecord,
    MemorySource,
)
from app.memory.ltm.manager import LTMManager


class FakeStore:
    def __init__(self) -> None:
        self.saved = []
        self.memories = {}

    def save(
        self,
        memory: MemoryRecord,
    ) -> None:
        self.saved.append(memory)
        self.memories[memory.memory_id] = memory

    def get(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> MemoryRecord | None:
        memory = self.memories.get(memory_id)

        if memory and memory.user_id == user_id:
            return memory

        return None


class FakeValidator:
    def validate(
        self,
        candidate: CandidateMemory,
    ) -> None:
        pass


class FakeRelatedRetriever:
    def __init__(
        self,
        memories: list[MemoryRecord],
    ) -> None:
        self.memories = memories

    def retrieve(
        self,
        candidate: CandidateMemory,
    ) -> list[MemoryRecord]:
        return self.memories


class FakeConsolidator:
    def __init__(
        self,
        decision: MemoryDecision,
    ) -> None:
        self.decision = decision

    def consolidate(
        self,
        *,
        candidate: CandidateMemory,
        related_memories: list[MemoryRecord],
    ) -> MemoryDecision:
        return self.decision


class FakeWriter:
    def __init__(self) -> None:
        self.written = []
        self.removed = []

    def write(
        self,
        memory: MemoryRecord,
    ) -> None:
        self.written.append(memory)

    def remove(
        self,
        memory_id: str,
    ) -> None:
        self.removed.append(memory_id)


def _candidate(
    content: str = "User prefers Qdrant.",
) -> CandidateMemory:
    return CandidateMemory(
        user_id="user_001",
        memory_type="preference",
        content=content,
        confidence=0.9,
        source=MemorySource(
            thread_id="thread_002",
            message_ids=["message_002"],
        ),
    )


def _memory(
    memory_id: str = "memory_001",
    content: str = "User prefers Chroma.",
    version: int = 1,
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
        version=version,
    )


def _manager(
    *,
    decision: MemoryDecision,
    related_memories: list[MemoryRecord],
):
    store = FakeStore()
    writer = FakeWriter()

    manager = LTMManager(
        memory_store=store,
        validator=FakeValidator(),
        related_retriever=FakeRelatedRetriever(
            related_memories
        ),
        consolidator=FakeConsolidator(
            decision
        ),
        writer=writer,
    )

    return manager, store, writer


def test_add_creates_and_indexes_memory() -> None:
    candidate = _candidate()

    decision = MemoryDecision(
        action="ADD",
        candidate=candidate,
    )

    manager, store, writer = _manager(
        decision=decision,
        related_memories=[],
    )

    result = manager.process(candidate)

    assert result == decision
    assert len(store.saved) == 1
    assert len(writer.written) == 1

    saved = store.saved[0]

    assert saved.user_id == "user_001"
    assert saved.content == "User prefers Qdrant."
    assert saved.status == "active"


def test_ignore_does_not_change_memory() -> None:
    candidate = _candidate()
    existing = _memory()

    decision = MemoryDecision(
        action="IGNORE",
        candidate=candidate,
        existing_memory_ids=[
            existing.memory_id
        ],
    )

    manager, store, writer = _manager(
        decision=decision,
        related_memories=[existing],
    )

    manager.process(candidate)

    assert store.saved == []
    assert writer.written == []
    assert writer.removed == []


def test_update_changes_existing_memory() -> None:
    candidate = _candidate(
        content="User prefers Qdrant."
    )
    existing = _memory(
        content="User prefers Chroma.",
        version=2,
    )

    decision = MemoryDecision(
        action="UPDATE",
        candidate=candidate,
        existing_memory_ids=[
            existing.memory_id
        ],
    )

    manager, store, writer = _manager(
        decision=decision,
        related_memories=[existing],
    )

    manager.process(candidate)

    assert len(store.saved) == 1
    assert len(writer.written) == 1

    updated = store.saved[0]

    assert updated.memory_id == existing.memory_id
    assert updated.content == (
        "User prefers Qdrant."
    )
    assert updated.version == 3
    assert updated.status == "active"


def test_supersede_creates_new_memory_and_deactivates_old() -> None:
    candidate = _candidate()
    existing = _memory(
        version=3,
    )

    decision = MemoryDecision(
        action="SUPERSEDE",
        candidate=candidate,
        existing_memory_ids=[
            existing.memory_id
        ],
    )

    manager, store, writer = _manager(
        decision=decision,
        related_memories=[existing],
    )

    manager.process(candidate)

    assert len(store.saved) == 2

    old_memory = store.saved[0]
    new_memory = store.saved[1]

    assert old_memory.memory_id == (
        existing.memory_id
    )
    assert old_memory.status == "superseded"

    assert new_memory.memory_id != (
        existing.memory_id
    )
    assert new_memory.status == "active"
    assert new_memory.version == 4
    assert new_memory.supersedes_memory_id == (
        existing.memory_id
    )

    assert writer.removed == [
        existing.memory_id
    ]


def test_merge_creates_combined_memory() -> None:
    candidate = _candidate(
        content="User builds backend systems."
    )

    first = _memory(
        memory_id="memory_001",
        content="User prefers Python.",
        version=2,
    )

    second = _memory(
        memory_id="memory_002",
        content="User works with APIs.",
        version=3,
    ).model_copy(
        update={
            "source": MemorySource(
                thread_id="thread_003",
                message_ids=["message_003"],
            )
        }
    )

    decision = MemoryDecision(
        action="MERGE",
        candidate=candidate,
        existing_memory_ids=[
            "memory_001",
            "memory_002",
        ],
    )

    manager, store, writer = _manager(
        decision=decision,
        related_memories=[
            first,
            second,
        ],
    )

    manager.process(candidate)

    assert len(store.saved) == 3

    merged = store.saved[-1]

    assert merged.status == "active"
    assert merged.version == 4
    assert "User prefers Python." in (
        merged.content
    )
    assert "User works with APIs." in (
        merged.content
    )
    assert "User builds backend systems." in (
        merged.content
    )

    assert writer.removed == [
        "memory_001",
        "memory_002",
    ]

    assert merged.metadata["source_history"] == [
        {
            "thread_id": "thread_001",
            "message_ids": ["message_001"],
        },
        {
            "thread_id": "thread_003",
            "message_ids": ["message_003"],
        },
        {
            "thread_id": "thread_002",
            "message_ids": ["message_002"],
        },
    ]