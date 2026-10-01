from app.contracts.memory import (
    CandidateMemory,
    MemoryRecord,
    MemorySource,
)
from app.contracts.retrieval import RetrievedItem
from app.memory.ltm.related import LTMRelatedMemoryRetriever


class FakeEmbeddings:
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(
        self,
        items: list[RetrievedItem],
    ) -> None:
        self.items = items
        self.filters = None
        self.collection = None
        self.top_k = None

    def search(
        self,
        *,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedItem]:
        self.collection = collection
        self.filters = filters
        self.top_k = top_k
        return self.items


class FakeMemoryStore:
    def __init__(
        self,
        memories: list[MemoryRecord],
    ) -> None:
        self.memories = {
            memory.memory_id: memory
            for memory in memories
        }

    def get(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> MemoryRecord | None:
        memory = self.memories.get(memory_id)

        if memory is None:
            return None

        if memory.user_id != user_id:
            return None

        return memory


def _candidate() -> CandidateMemory:
    return CandidateMemory(
        user_id="user_001",
        memory_type="preference",
        content="User prefers concise responses.",
        confidence=0.9,
        source=MemorySource(
            thread_id="thread_002",
            message_ids=["message_002"],
        ),
    )


def _memory(
    memory_id: str = "memory_001",
    status: str = "active",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="user_001",
        memory_type="preference",
        content="User prefers concise responses.",
        status=status,
        source=MemorySource(
            thread_id="thread_001",
            message_ids=["message_001"],
        ),
    )


def test_related_retriever_returns_structured_memories() -> None:
    memory = _memory()

    vector_store = FakeVectorStore(
        [
            RetrievedItem(
                item_id="memory_001",
                source="ltm",
                content=memory.content,
                score=0.95,
                rank=1,
                metadata={},
            )
        ]
    )

    retriever = LTMRelatedMemoryRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        memory_store=FakeMemoryStore([memory]),
    )

    result = retriever.retrieve(
        _candidate()
    )

    assert result == [memory]


def test_related_retriever_filters_by_user_and_status() -> None:
    vector_store = FakeVectorStore([])

    retriever = LTMRelatedMemoryRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        memory_store=FakeMemoryStore([]),
    )

    retriever.retrieve(_candidate())

    assert vector_store.collection == "ltm_collection"
    assert vector_store.filters == {
        "user_id": "user_001",
        "status": "active",
    }


def test_related_retriever_respects_top_k() -> None:
    vector_store = FakeVectorStore([])

    retriever = LTMRelatedMemoryRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        memory_store=FakeMemoryStore([]),
    )

    retriever.retrieve(
        _candidate(),
        top_k=3,
    )

    assert vector_store.top_k == 3


def test_related_retriever_skips_missing_memory() -> None:
    vector_store = FakeVectorStore(
        [
            RetrievedItem(
                item_id="missing",
                source="ltm",
                content="Missing memory",
                score=0.9,
                rank=1,
                metadata={},
            )
        ]
    )

    retriever = LTMRelatedMemoryRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        memory_store=FakeMemoryStore([]),
    )

    result = retriever.retrieve(
        _candidate()
    )

    assert result == []


def test_related_retriever_skips_inactive_memory() -> None:
    memory = _memory(
        status="inactive"
    )

    vector_store = FakeVectorStore(
        [
            RetrievedItem(
                item_id=memory.memory_id,
                source="ltm",
                content=memory.content,
                score=0.9,
                rank=1,
                metadata={},
            )
        ]
    )

    retriever = LTMRelatedMemoryRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        memory_store=FakeMemoryStore([memory]),
    )

    result = retriever.retrieve(
        _candidate()
    )

    assert result == []