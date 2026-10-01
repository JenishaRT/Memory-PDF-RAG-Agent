from datetime import datetime, timezone

from app.memory.stm.episodic_retriever import STMEpisodicRetriever
from app.memory.stm.models import STMMessage
from app.memory.stm.writer import STMWriter
from app.vectorstores.chroma import ChromaVectorStore


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_episodic_retriever_filters_date_across_threads(tmp_path) -> None:
    vector_store = ChromaVectorStore(str(tmp_path / "chroma"))
    writer = STMWriter(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )

    messages = [
        STMMessage(
            message_id="message_001",
            user_id="user_001",
            thread_id="thread_001",
            role="user",
            content="Discussed a project.",
            timestamp=datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc),
        ),
        STMMessage(
            message_id="message_002",
            user_id="user_001",
            thread_id="thread_002",
            role="assistant",
            content="Completed a report.",
            timestamp=datetime(2026, 9, 18, 15, 0, tzinfo=timezone.utc),
        ),
        STMMessage(
            message_id="message_003",
            user_id="user_001",
            thread_id="thread_001",
            role="user",
            content="This was the day before.",
            timestamp=datetime(2026, 9, 17, 23, 59, tzinfo=timezone.utc),
        ),
        STMMessage(
            message_id="message_004",
            user_id="user_002",
            thread_id="thread_003",
            role="user",
            content="Another user's message.",
            timestamp=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        ),
    ]

    for message in messages:
        writer.write(message)

    result = STMEpisodicRetriever(
        vector_store=vector_store,
    ).retrieve(
        user_id="user_001",
        query="What did I do on September 18?",
        start_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        end_at=datetime(2026, 9, 19, tzinfo=timezone.utc),
    )

    assert [item.item_id for item in result.items] == [
        "message_001",
        "message_002",
    ]
    assert [item.metadata["thread_id"] for item in result.items] == [
        "thread_001",
        "thread_002",
    ]