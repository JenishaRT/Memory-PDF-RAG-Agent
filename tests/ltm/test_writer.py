from app.contracts.memory import MemoryRecord, MemorySource
from app.memory.ltm.writer import LTMWriter


class FakeEmbeddings:
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.add_calls = []
        self.delete_calls = []

    def add(
        self,
        *,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self.add_calls.append(
            {
                "collection": collection,
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

    def delete(
        self,
        *,
        collection: str,
        ids: list[str],
    ) -> None:
        self.delete_calls.append(
            {
                "collection": collection,
                "ids": ids,
            }
        )


def _memory(
    *,
    status: str = "active",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory_001",
        user_id="user_001",
        memory_type="preference",
        content="User prefers concise responses.",
        status=status,
        source=MemorySource(
            thread_id="thread_001",
            message_ids=["message_001"],
        ),
    )


def test_writer_indexes_memory() -> None:
    vector_store = FakeVectorStore()

    writer = LTMWriter(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    writer.write(_memory())

    assert len(vector_store.add_calls) == 1

    call = vector_store.add_calls[0]

    assert call["collection"] == "ltm_collection"
    assert call["ids"] == ["memory_001"]
    assert call["documents"] == [
        "User prefers concise responses."
    ]
    assert call["embeddings"] == [
        [0.1, 0.2, 0.3]
    ]

    assert call["metadatas"][0] == {
        "source": "ltm",
        "user_id": "user_001",
        "memory_type": "preference",
        "status": "active",
        "thread_id": "thread_001",
        "version": 1,
    }


def test_writer_indexes_memory_status() -> None:
    vector_store = FakeVectorStore()

    writer = LTMWriter(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    writer.write(
        _memory(status="superseded")
    )

    assert (
        vector_store.add_calls[0]["metadatas"][0]["status"]
        == "superseded"
    )


def test_writer_removes_memory_from_index() -> None:
    vector_store = FakeVectorStore()

    writer = LTMWriter(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    writer.remove("memory_001")

    assert vector_store.delete_calls == [
        {
            "collection": "ltm_collection",
            "ids": ["memory_001"],
        }
    ]


def test_writer_rejects_empty_memory() -> None:
    vector_store = FakeVectorStore()

    writer = LTMWriter(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    memory = _memory().model_copy(
        update={"content": " "}
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="empty memory",
    ):
        writer.write(memory)

    assert vector_store.add_calls == []