from app.contracts.memory import MemoryRecord
from app.embeddings.provider import EmbeddingProvider
from app.vectorstores.base import VectorStore


class LTMWriter:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    def write(
        self,
        memory: MemoryRecord,
    ) -> None:
        if not memory.content.strip():
            raise ValueError(
                "Cannot index an empty memory"
            )

        embedding = self.embeddings.embed_documents(
            [memory.content]
        )[0]

        self.vector_store.add(
            collection="ltm_collection",
            ids=[memory.memory_id],
            documents=[memory.content],
            embeddings=[embedding],
            metadatas=[
                {
                    "source": "ltm",
                    "user_id": memory.user_id,
                    "memory_type": memory.memory_type,
                    "status": memory.status,
                    "thread_id": memory.source.thread_id,
                    "version": memory.version,
                }
            ],
        )

    def remove(
        self,
        memory_id: str,
    ) -> None:
        self.vector_store.delete(
            collection="ltm_collection",
            ids=[memory_id],
        )