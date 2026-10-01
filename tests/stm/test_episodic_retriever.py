from datetime import datetime, timezone

import pytest

from app.contracts.errors import RetrievalError
from app.contracts.retrieval import RetrievedItem
from app.memory.stm.episodic_retriever import STMEpisodicRetriever


class FakeVectorStore:
    def __init__(self, items: list[RetrievedItem]) -> None:
        self.items = items
        self.collection = None
        self.filters = None

    def get_by_metadata(self, *, collection, filters):
        self.collection = collection
        self.filters = filters
        return self.items


def _item(item_id: str, thread_id: str, timestamp: str) -> RetrievedItem:
    return RetrievedItem(
        item_id=item_id,
        source="stm",
        content=f"Message {item_id}",
        metadata={
            "user_id": "user_001",
            "thread_id": thread_id,
            "timestamp": timestamp,
        },
    )


def test_retrieves_all_threads_for_user_date_in_timestamp_order() -> None:
    store = FakeVectorStore([
        _item("later", "thread_002", "2026-09-18T15:00:00.000000+00:00"),
        _item("earlier", "thread_001", "2026-09-18T09:00:00.000000+00:00"),
    ])
    retriever = STMEpisodicRetriever(vector_store=store)

    result = retriever.retrieve(
        user_id="user_001",
        query="What did I do on September 18?",
        start_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        end_at=datetime(2026, 9, 19, tzinfo=timezone.utc),
    )

    assert store.collection == "stm_collection"
    assert store.filters == {
        "user_id": "user_001",
        "timestamp_epoch": {
            "$gte": 1789689600.0,
            "$lt": 1789776000.0,
        },
    }
    assert [item.item_id for item in result.items] == ["earlier", "later"]
    assert [item.rank for item in result.items] == [1, 2]
    assert result.metadata["cross_thread"] is True


def test_rejects_naive_date_boundaries() -> None:
    retriever = STMEpisodicRetriever(vector_store=FakeVectorStore([]))

    with pytest.raises(RetrievalError, match="timezone"):
        retriever.retrieve(
            user_id="user_001",
            query="What did I do?",
            start_at=datetime(2026, 9, 18),
            end_at=datetime(2026, 9, 19, tzinfo=timezone.utc),
        )


def test_rejects_end_at_not_after_start_at() -> None:
    retriever = STMEpisodicRetriever(vector_store=FakeVectorStore([]))
    boundary = datetime(2026, 9, 18, tzinfo=timezone.utc)

    with pytest.raises(RetrievalError, match="after"):
        retriever.retrieve(
            user_id="user_001",
            query="What did I do?",
            start_at=boundary,
            end_at=boundary,
        )