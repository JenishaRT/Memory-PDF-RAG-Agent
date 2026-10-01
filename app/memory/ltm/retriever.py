import time

from app.contracts.errors import RetrievalError
from app.contracts.memory import MemoryRecord
from app.contracts.retrieval import LTMQuery, RetrievalResult, RetrievedItem
from app.embeddings.provider import EmbeddingProvider
from app.vectorstores.base import VectorStore


class LTMRetriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(
        self,
        query: LTMQuery,
    ) -> RetrievalResult:
        if not query.query.strip():
            raise RetrievalError(
                "Cannot retrieve LTM with an empty query"
            )

        started_at = time.perf_counter()

        try:
            query_embedding = self.embeddings.embed_query(
                query.query
            )

            items = self.vector_store.search(
                collection="ltm_collection",
                query_embedding=query_embedding,
                top_k=query.top_k,
                filters={
                    "user_id": query.user_id,
                    "status": "active",
                },
            )

            latency_ms = (
                time.perf_counter() - started_at
            ) * 1000

            return RetrievalResult(
                source="ltm",
                query=query.query,
                items=items,
                latency_ms=latency_ms,
                metadata={
                    "user_id": query.user_id,
                    "top_k": query.top_k,
                    "active_only": True,
                },
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                "Failed to retrieve LTM"
            ) from exc