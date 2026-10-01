from datetime import datetime, timezone
import math
import time

from app.contracts.errors import RetrievalError
from app.contracts.retrieval import RetrievalResult, RetrievedItem, STMQuery
from app.embeddings.provider import EmbeddingProvider
from app.vectorstores.base import VectorStore


class STMRetriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(self, query: STMQuery) -> RetrievalResult:
        if not query.query.strip():
            raise RetrievalError(
                "Cannot retrieve STM with an empty query"
            )

        started_at = time.perf_counter()

        try:
            query_embedding = self.embeddings.embed_query(
                query.query
            )

            items = self.vector_store.search(
                collection="stm_collection",
                query_embedding=query_embedding,
                top_k=query.top_k,
                filters={
                    "user_id": query.user_id,
                    "thread_id": query.thread_id,
                },
            )

            if query.recency_weight > 0:
                items = self._apply_recency_weight(
                    items,
                    query.recency_weight,
                )

            latency_ms = (
                time.perf_counter() - started_at
            ) * 1000

            return RetrievalResult(
                source="stm",
                query=query.query,
                items=items,
                latency_ms=latency_ms,
                metadata={
                    "user_id": query.user_id,
                    "thread_id": query.thread_id,
                    "recency_weight": query.recency_weight,
                },
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                "Failed to retrieve STM"
            ) from exc

    def _apply_recency_weight(
        self,
        items: list[RetrievedItem],
        recency_weight: float,
    ) -> list[RetrievedItem]:
        now = datetime.now(timezone.utc)
        scored_items: list[tuple[float, RetrievedItem]] = []

        for item in items:
            semantic_score = item.score or 0.0
            recency_score = self._recency_score(
                item.metadata.get("timestamp"),
                now,
            )

            combined_score = (
                semantic_score * (1.0 - recency_weight)
                + recency_score * recency_weight
            )

            updated_item = item.model_copy(
                update={"score": combined_score}
            )

            scored_items.append(
                (combined_score, updated_item)
            )

        scored_items.sort(
            key=lambda value: value[0],
            reverse=True,
        )

        results: list[RetrievedItem] = []

        for rank, (_, item) in enumerate(scored_items, start=1):
            results.append(
                item.model_copy(
                    update={"rank": rank}
                )
            )

        return results

    @staticmethod
    def _recency_score(
        timestamp: object,
        now: datetime,
    ) -> float:
        if not timestamp:
            return 0.0

        try:
            message_time = datetime.fromisoformat(
                str(timestamp)
            )

            if message_time.tzinfo is None:
                message_time = message_time.replace(
                    tzinfo=timezone.utc
                )

            age_hours = max(
                0.0,
                (now - message_time).total_seconds() / 3600,
            )

            return math.exp(-age_hours / 24.0)
        except (TypeError, ValueError):
            return 0.0