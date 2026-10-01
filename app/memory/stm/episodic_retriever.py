from datetime import datetime, timezone

from app.contracts.errors import RetrievalError
from app.contracts.retrieval import RetrievalResult
from app.vectorstores.chroma import ChromaVectorStore


class STMEpisodicRetriever:
    def __init__(self, *, vector_store: ChromaVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        start_at: datetime,
        end_at: datetime,
    ) -> RetrievalResult:
        if not user_id.strip() or not query.strip():
            raise RetrievalError("user_id and query must not be empty")

        if (
            start_at.tzinfo is None
            or start_at.utcoffset() is None
            or end_at.tzinfo is None
            or end_at.utcoffset() is None
        ):
            raise RetrievalError("Date range must include timezone information")

        start = start_at.astimezone(timezone.utc)
        end = end_at.astimezone(timezone.utc)
        if end <= start:
            raise RetrievalError("end_at must be after start_at")

        items = self.vector_store.get_by_metadata(
            collection="stm_collection",
            filters={
                "user_id": user_id,
                "timestamp_epoch": {
                    "$gte": start.timestamp(),
                    "$lt": end.timestamp(),
                },
            },
        )

        items.sort(key=lambda item: item.metadata.get("timestamp", ""))

        return RetrievalResult(
            source="stm",
            query=query,
            items=[
                item.model_copy(update={"rank": rank})
                for rank, item in enumerate(items, start=1)
            ],
            metadata={
                "user_id": user_id,
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "cross_thread": True,
            },
        )