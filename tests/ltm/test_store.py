from app.contracts.memory import MemoryRecord, MemorySource
from app.memory.ltm.store import JsonMemoryStore


def _memory(
    *,
    memory_id: str = "memory_001",
    user_id: str = "user_001",
    content: str = "User prefers Chroma.",
    status: str = "active",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id=user_id,
        memory_type="preference",
        content=content,
        status=status,
        source=MemorySource(
            thread_id="thread_001",
            message_ids=["msg_001"],
        ),
    )


def test_store_saves_and_gets_memory(tmp_path) -> None:
    store = JsonMemoryStore(
        tmp_path / "memories.json"
    )

    memory = _memory()

    store.save(memory)

    result = store.get(
        user_id="user_001",
        memory_id="memory_001",
    )

    assert result == memory


def test_store_returns_only_active_memories_by_default(
    tmp_path,
) -> None:
    store = JsonMemoryStore(
        tmp_path / "memories.json"
    )

    active = _memory(
        memory_id="memory_001",
        status="active",
    )
    inactive = _memory(
        memory_id="memory_002",
        status="inactive",
    )

    store.save(active)
    store.save(inactive)

    result = store.list(
        user_id="user_001",
    )

    assert result == [active]


def test_store_can_include_inactive_memories(
    tmp_path,
) -> None:
    store = JsonMemoryStore(
        tmp_path / "memories.json"
    )

    active = _memory(
        memory_id="memory_001",
        status="active",
    )
    inactive = _memory(
        memory_id="memory_002",
        status="inactive",
    )

    store.save(active)
    store.save(inactive)

    result = store.list(
        user_id="user_001",
        include_inactive=True,
    )

    assert result == [active, inactive]


def test_store_isolates_users(tmp_path) -> None:
    store = JsonMemoryStore(
        tmp_path / "memories.json"
    )

    user_one = _memory(
        memory_id="memory_001",
        user_id="user_001",
    )
    user_two = _memory(
        memory_id="memory_002",
        user_id="user_002",
    )

    store.save(user_one)
    store.save(user_two)

    result = store.list(
        user_id="user_001",
    )

    assert result == [user_one]


def test_store_updates_existing_memory(
    tmp_path,
) -> None:
    store = JsonMemoryStore(
        tmp_path / "memories.json"
    )

    memory = _memory()

    store.save(memory)

    updated = memory.model_copy(
        update={
            "content": "User prefers Qdrant.",
            "version": 2,
        }
    )

    store.save(updated)

    result = store.list(
        user_id="user_001",
        include_inactive=True,
    )

    assert result == [updated]


def test_store_persists_across_instances(
    tmp_path,
) -> None:
    path = tmp_path / "memories.json"

    first_store = JsonMemoryStore(path)
    memory = _memory()

    first_store.save(memory)

    second_store = JsonMemoryStore(path)

    result = second_store.get(
        user_id="user_001",
        memory_id="memory_001",
    )

    assert result == memory


def test_store_does_not_return_another_users_memory(
    tmp_path,
) -> None:
    store = JsonMemoryStore(
        tmp_path / "memories.json"
    )

    memory = _memory(
        user_id="user_001",
    )

    store.save(memory)

    result = store.get(
        user_id="user_002",
        memory_id="memory_001",
    )

    assert result is None