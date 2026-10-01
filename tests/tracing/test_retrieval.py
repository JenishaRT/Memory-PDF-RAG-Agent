from app.contracts.retrieval import RetrievalResult, RetrievedItem, STMQuery
from app.tracing.retrieval import STMTraceBuilder


def test_stm_trace_contains_retrieval_details() -> None:
    query = STMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="What were we discussing?",
        top_k=3,
        recency_weight=0.25,
        include_recent_messages=True,
    )

    result = RetrievalResult(
        source="stm",
        query=query.query,
        items=[
            RetrievedItem(
                item_id="msg_1",
                source="stm",
                content="We discussed the PDF agent.",
                score=0.91,
                rank=1,
                metadata={
                    "user_id": "user_001",
                    "thread_id": "thread_001",
                },
            ),
            RetrievedItem(
                item_id="msg_2",
                source="stm",
                content="We discussed OCR.",
                score=0.82,
                rank=2,
                metadata={
                    "user_id": "user_001",
                    "thread_id": "thread_001",
                },
            ),
        ],
        latency_ms=12.5,
    )

    trace = STMTraceBuilder().build(
        query=query,
        result=result,
    )

    assert trace.source == "stm"
    assert trace.query == "What were we discussing?"
    assert trace.retrieved_ids == ["msg_1", "msg_2"]
    assert trace.scores == [0.91, 0.82]
    assert trace.ranks == [1, 2]
    assert trace.latency_ms == 12.5
    assert trace.metadata["user_id"] == "user_001"
    assert trace.metadata["thread_id"] == "thread_001"
    assert trace.metadata["top_k"] == 3
    assert trace.metadata["recency_weight"] == 0.25
    assert trace.metadata["include_recent_messages"] is True


def test_stm_trace_handles_empty_results() -> None:
    query = STMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="Nothing relevant",
        top_k=3,
    )

    result = RetrievalResult(
        source="stm",
        query=query.query,
        items=[],
        latency_ms=4.2,
    )

    trace = STMTraceBuilder().build(
        query=query,
        result=result,
    )

    assert trace.source == "stm"
    assert trace.retrieved_ids == []
    assert trace.scores == []
    assert trace.ranks == []
    assert trace.latency_ms == 4.2