from unittest.mock import Mock

import pytest

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from app.contracts.errors import RetrievalError
from app.contracts.retrieval import RetrievedItem, STMQuery
from app.memory.stm.retriever import STMRetriever


def test_stm_retriever() -> None:
    embeddings = Mock()
    vector_store = Mock()

    embeddings.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    vector_store.search.return_value = [
        RetrievedItem(
            item_id="message_001",
            source="stm",
            content="I prefer Python.",
            score=0.95,
            rank=1,
            metadata={
                "user_id": "user_001",
                "thread_id": "thread_001",
            },
        )
    ]

    retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = retriever.retrieve(
        STMQuery(
            user_id="user_001",
            thread_id="thread_001",
            query="What programming language do I prefer?",
            top_k=5,
        )
    )

    assert result.source == "stm"
    assert result.query == (
        "What programming language do I prefer?"
    )
    assert len(result.items) == 1
    assert result.items[0].item_id == "message_001"

    embeddings.embed_query.assert_called_once_with(
        "What programming language do I prefer?"
    )

    vector_store.search.assert_called_once_with(
        collection="stm_collection",
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
        filters={
            "user_id": "user_001",
            "thread_id": "thread_001",
        },
    )


def test_stm_retriever_does_not_cross_threads() -> None:
    embeddings = Mock()
    vector_store = Mock()

    vector_store.search.return_value = [
        RetrievedItem(
            item_id="message_001",
            source="stm",
            content="Message from thread 001.",
            score=0.95,
            rank=1,
            metadata={
                "user_id": "user_001",
                "thread_id": "thread_001",
            },
        )
    ]

    retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = retriever.retrieve(
        STMQuery(
            user_id="user_001",
            thread_id="thread_001",
            query="What did I say?",
        )
    )

    assert all(
        item.metadata["thread_id"] == "thread_001"
        for item in result.items
    )

    filters = vector_store.search.call_args.kwargs["filters"]

    assert filters["user_id"] == "user_001"
    assert filters["thread_id"] == "thread_001"


def test_stm_retriever_does_not_cross_users() -> None:
    embeddings = Mock()
    vector_store = Mock()

    vector_store.search.return_value = [
        RetrievedItem(
            item_id="message_001",
            source="stm",
            content="User 001 message.",
            score=0.95,
            rank=1,
            metadata={
                "user_id": "user_001",
                "thread_id": "thread_001",
            },
        )
    ]

    retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    retriever.retrieve(
        STMQuery(
            user_id="user_001",
            thread_id="thread_001",
            query="What did I say?",
        )
    )

    filters = vector_store.search.call_args.kwargs["filters"]

    assert filters["user_id"] == "user_001"
    assert filters["thread_id"] == "thread_001"


def test_stm_retriever_rejects_empty_query() -> None:
    embeddings = Mock()
    vector_store = Mock()

    retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    with pytest.raises(RetrievalError):
        retriever.retrieve(
            STMQuery(
                user_id="user_001",
                thread_id="thread_001",
                query="   ",
            )
        )

    embeddings.embed_query.assert_not_called()
    vector_store.search.assert_not_called()

def test_stm_recency_weighting_prefers_recent_message() -> None:
    embeddings = Mock()
    vector_store = Mock()

    now = datetime.now(timezone.utc)

    vector_store.search.return_value = [
        RetrievedItem(
            item_id="old_message",
            source="stm",
            content="Older message.",
            score=0.9,
            rank=1,
            metadata={
                "timestamp": (
                    now - timedelta(days=7)
                ).isoformat(),
            },
        ),
        RetrievedItem(
            item_id="recent_message",
            source="stm",
            content="Recent message.",
            score=0.8,
            rank=2,
            metadata={
                "timestamp": now.isoformat(),
            },
        ),
    ]

    retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = retriever.retrieve(
        STMQuery(
            user_id="user_001",
            thread_id="thread_001",
            query="What did we discuss?",
            top_k=2,
            recency_weight=0.8,
        )
    )

    assert result.items[0].item_id == "recent_message"
    assert result.items[0].rank == 1
    assert result.items[1].rank == 2

def test_stm_recency_weight_zero_keeps_semantic_scores() -> None:
    embeddings = Mock()
    vector_store = Mock()

    now = datetime.now(timezone.utc)

    vector_store.search.return_value = [
        RetrievedItem(
            item_id="message_001",
            source="stm",
            content="Highly relevant older message.",
            score=0.95,
            rank=1,
            metadata={
                "timestamp": (
                    now - timedelta(days=7)
                ).isoformat(),
            },
        ),
        RetrievedItem(
            item_id="message_002",
            source="stm",
            content="Less relevant recent message.",
            score=0.80,
            rank=2,
            metadata={
                "timestamp": now.isoformat(),
            },
        ),
    ]

    retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = retriever.retrieve(
        STMQuery(
            user_id="user_001",
            thread_id="thread_001",
            query="What did we discuss?",
            top_k=2,
            recency_weight=0.0,
        )
    )

    assert result.items[0].item_id == "message_001"
    assert result.items[0].score == 0.95
    assert result.items[1].item_id == "message_002"
    assert result.items[1].score == 0.80

def test_stm_invalid_timestamp_does_not_break_retrieval() -> None:
    embeddings = Mock()
    vector_store = Mock()

    vector_store.search.return_value = [
        RetrievedItem(
            item_id="message_001",
            source="stm",
            content="Some message.",
            score=0.9,
            rank=1,
            metadata={
                "timestamp": "invalid-timestamp",
            },
        )
    ]

    retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = retriever.retrieve(
        STMQuery(
            user_id="user_001",
            thread_id="thread_001",
            query="What did we discuss?",
            top_k=1,
            recency_weight=0.5,
        )
    )

    assert len(result.items) == 1
    assert result.items[0].item_id == "message_001"
    