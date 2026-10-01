from app.contracts.memory import (
    CandidateMemory,
    MemoryDecision,
    MemorySource,
)
from app.contracts.retrieval import (
    LTMQuery,
    RetrievalResult,
    RetrievedItem,
)
from app.tracing.ltm import LTMTraceBuilder


def test_ltm_retrieval_trace():
    query = LTMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="What database does the user prefer?",
        top_k=5,
    )

    result = RetrievalResult(
        source="ltm",
        query=query.query,
        items=[
            RetrievedItem(
                item_id="memory_001",
                source="ltm",
                content="User prefers Chroma.",
                score=0.91,
                rank=1,
                metadata={
                    "user_id": "user_001",
                    "status": "active",
                },
            ),
        ],
        latency_ms=12.5,
        metadata={
            "user_id": "user_001",
            "top_k": 5,
        },
    )

    trace = LTMTraceBuilder().build_retrieval(
        query=query,
        result=result,
    )

    assert trace.source == "ltm"
    assert trace.query == query.query
    assert trace.retrieved_ids == ["memory_001"]
    assert trace.scores == [0.91]
    assert trace.ranks == [1]
    assert trace.latency_ms == 12.5
    assert trace.metadata["user_id"] == "user_001"
    assert trace.metadata["thread_id"] == "thread_001"
    assert trace.metadata["top_k"] == 5


def test_ltm_memory_operation_trace():
    candidate = CandidateMemory(
        user_id="user_001",
        memory_type="preference",
        content="User now prefers Qdrant.",
        confidence=0.96,
        source=MemorySource(
            thread_id="thread_002",
            message_ids=["message_002"],
        ),
    )

    decision = MemoryDecision(
        action="SUPERSEDE",
        candidate=candidate,
        existing_memory_ids=["memory_001"],
        reason="The user changed their preference.",
    )

    trace = LTMTraceBuilder().build_memory_operation(
        candidate=candidate,
        decision=decision,
        latency_ms=8.4,
    )

    assert trace.operation == "consolidate"
    assert trace.user_id == "user_001"
    assert trace.thread_id == "thread_002"
    assert trace.memory_ids == ["memory_001"]
    assert trace.action == "SUPERSEDE"
    assert trace.latency_ms == 8.4
    assert trace.metadata["memory_type"] == "preference"
    assert trace.metadata["confidence"] == 0.96
    assert trace.metadata["source_message_ids"] == ["message_002"]