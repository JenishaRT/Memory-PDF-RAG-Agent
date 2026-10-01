from app.contracts.memory import (
    CandidateMemory,
    MemoryDecision,
    MemorySource,
)
from app.contracts.retrieval import LTMQuery, RetrievedItem, RetrievalResult
from app.tracing.retrieval import LTMTraceBuilder


def test_ltm_retrieval_trace():
    query = LTMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="What database does the user prefer?",
        top_k=3,
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
                metadata={"status": "active"},
            ),
            RetrievedItem(
                item_id="memory_002",
                source="ltm",
                content="User works with vector databases.",
                score=0.82,
                rank=2,
                metadata={"status": "active"},
            ),
        ],
        latency_ms=12.5,
        metadata={},
    )

    trace = LTMTraceBuilder().build_retrieval(
        query=query,
        result=result,
    )

    assert trace.source == "ltm"
    assert trace.query == "What database does the user prefer?"
    assert trace.retrieved_ids == [
        "memory_001",
        "memory_002",
    ]
    assert trace.scores == [0.91, 0.82]
    assert trace.ranks == [1, 2]
    assert trace.latency_ms == 12.5
    assert trace.metadata["user_id"] == "user_001"
    assert trace.metadata["top_k"] == 3
    assert trace.metadata["active_only"] is True


def test_ltm_consolidation_trace():
    candidate = CandidateMemory(
        user_id="user_001",
        memory_type="preference",
        content="User now prefers Qdrant.",
        confidence=0.95,
        source=MemorySource(
            thread_id="thread_001",
            message_ids=["message_010"],
        ),
    )

    decision = MemoryDecision(
        action="SUPERSEDE",
        candidate=candidate,
        existing_memory_ids=["memory_001"],
        reason="The new preference replaces the previous preference.",
    )

    trace = LTMTraceBuilder().build_consolidation(
        candidate=candidate,
        decision=decision,
        latency_ms=18.2,
    )

    assert trace.operation == "consolidate"
    assert trace.user_id == "user_001"
    assert trace.thread_id == "thread_001"
    assert trace.memory_ids == ["memory_001"]
    assert trace.action == "SUPERSEDE"
    assert trace.latency_ms == 18.2
    assert trace.metadata["memory_type"] == "preference"
    assert trace.metadata["source_message_ids"] == ["message_010"]
    assert trace.metadata["reason"] == (
        "The new preference replaces the previous preference."
    )