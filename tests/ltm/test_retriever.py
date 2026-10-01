from app.contracts.memory import MemoryRecord, MemorySource
from app.contracts.retrieval import LTMQuery
from app.memory.ltm.retriever import LTMRetriever


class FakeEmbeddings:
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self) -> None:
        self.collection = None
        self.filters = None
        self.top_k = None

    def search(
        self,
        *,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list:
        self.collection = collection
        self.filters = filters
        self.top_k = top_k

        return []


def test_retriever_uses_ltm_collection() -> None:
    vector_store = FakeVectorStore()

    retriever = LTMRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    query = LTMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="What database do I prefer?",
        top_k=5,
    )

    result = retriever.retrieve(query)

    assert result.source == "ltm"
    assert result.query == "What database do I prefer?"
    assert result.items == []

    assert vector_store.collection == "ltm_collection"
    assert vector_store.top_k == 5


def test_retriever_filters_by_user_and_active_status() -> None:
    vector_store = FakeVectorStore()

    retriever = LTMRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    query = LTMQuery(
        user_id="user_001",
        thread_id="thread_123",
        query="What database do I prefer?",
    )

    retriever.retrieve(query)

    assert vector_store.filters == {
        "user_id": "user_001",
        "status": "active",
    }


def test_retriever_does_not_filter_by_thread() -> None:
    vector_store = FakeVectorStore()

    retriever = LTMRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    query = LTMQuery(
        user_id="user_001",
        thread_id="thread_999",
        query="What are my preferences?",
    )

    retriever.retrieve(query)

    assert "thread_id" not in vector_store.filters


def test_retriever_rejects_empty_query() -> None:
    import pytest

    vector_store = FakeVectorStore()

    retriever = LTMRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    query = LTMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query=" ",
    )

    with pytest.raises(
        Exception,
        match="empty query",
    ):
        retriever.retrieve(query)