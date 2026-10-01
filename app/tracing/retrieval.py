from app.contracts.memory import CandidateMemory, MemoryDecision
from app.contracts.retrieval import LTMQuery, RetrievalResult
from app.contracts.tracing import MemoryTrace, RetrievalTrace


class STMTraceBuilder:
    def build(self, *, query, result: RetrievalResult) -> RetrievalTrace:
        return RetrievalTrace(
            source="stm",
            query=result.query,
            retrieved_ids=[item.item_id for item in result.items],
            scores=[item.score for item in result.items],
            ranks=[item.rank for item in result.items],
            latency_ms=result.latency_ms,
            metadata={
                "user_id": query.user_id,
                "thread_id": query.thread_id,
                "top_k": query.top_k,
                "recency_weight": query.recency_weight,
                "include_recent_messages": query.include_recent_messages,
            },
        )


class LTMTraceBuilder:
    def build_retrieval(
        self,
        *,
        query: LTMQuery,
        result: RetrievalResult,
    ) -> RetrievalTrace:
        return RetrievalTrace(
            source="ltm",
            query=result.query,
            retrieved_ids=[item.item_id for item in result.items],
            scores=[item.score for item in result.items],
            ranks=[item.rank for item in result.items],
            latency_ms=result.latency_ms,
            metadata={
                "user_id": query.user_id,
                "top_k": query.top_k,
                "active_only": True,
            },
        )

    def build_consolidation(
        self,
        *,
        candidate: CandidateMemory,
        decision: MemoryDecision,
        latency_ms: float | None = None,
    ) -> MemoryTrace:
        return MemoryTrace(
            operation="consolidate",
            user_id=candidate.user_id,
            thread_id=candidate.source.thread_id,
            memory_ids=decision.existing_memory_ids,
            action=decision.action,
            latency_ms=latency_ms,
            metadata={
                "memory_type": candidate.memory_type,
                "candidate_content": candidate.content,
                "source_message_ids": candidate.source.message_ids,
                "reason": decision.reason,
            },
        )