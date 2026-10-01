from app.contracts.memory import CandidateMemory, MemoryRecord
from app.embeddings.provider import EmbeddingProvider
from app.vectorstores.base import VectorStore


class LTMRelatedMemoryRetriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        memory_store,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.memory_store = memory_store

    def retrieve(
        self,
        candidate: CandidateMemory,
        *,
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        if top_k <= 0:
            return []

        embedding = self.embeddings.embed_query(
            candidate.content
        )

        items = self.vector_store.search(
            collection="ltm_collection",
            query_embedding=embedding,
            top_k=top_k,
            filters={
                "user_id": candidate.user_id,
                "status": "active",
            },
        )

        memories: list[MemoryRecord] = []

        for item in items:
            memory = self.memory_store.get(
                user_id=candidate.user_id,
                memory_id=item.item_id,
            )

            if memory is None:
                continue

            if memory.status != "active":
                continue

            memories.append(memory)

        return memories