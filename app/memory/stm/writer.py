from app.contracts.errors import MemoryError
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from app.memory.stm.models import STMMessage
from app.vectorstores.chroma import ChromaVectorStore

from datetime import timezone

class STMWriter:
    def __init__(
        self,
        *,
        embeddings: SentenceTransformerEmbeddings,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    def write(self, message: STMMessage) -> None:
        if not message.content.strip():
            raise MemoryError(
                "Cannot store an empty STM message"
            )

        try:
            embedding = self.embeddings.embed_documents(
                [message.content]
            )[0]

            timestamp = message.timestamp
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)

            metadata = dict(message.metadata)
            metadata.update({
                "source": "stm",
                "user_id": message.user_id,
                "thread_id": message.thread_id,
                "role": message.role,
                "timestamp": timestamp.isoformat(timespec="microseconds"),
                "timestamp_epoch": timestamp.timestamp(),
            })

            self.vector_store.add(
                collection="stm_collection",
                ids=[message.message_id],
                documents=[message.content],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError(
                f"Failed to store STM message: {message.message_id}"
            ) from exc