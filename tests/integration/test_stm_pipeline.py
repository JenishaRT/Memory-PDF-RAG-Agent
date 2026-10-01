from datetime import datetime, timezone

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.retrieval import STMQuery, RetrievedItem
from app.llm.provider import ChatMessage, LLMResponse
from app.memory.stm.context import STMContextExpander
from app.memory.stm.context_budget import STMContextBudget
from app.memory.stm.manager import STMManager
from app.memory.stm.retriever import STMRetriever
from app.memory.stm.summarizer import STMContextSummarizer


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def search(
        self,
        *,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedItem]:
        assert collection == "stm_collection"

        if filters != {
            "user_id": "user_001",
            "thread_id": "thread_001",
        }:
            return []

        return [
            RetrievedItem(
                item_id="msg_2",
                source="stm",
                content="We discussed the PDF agent.",
                score=0.95,
                rank=1,
                metadata={
                    "user_id": "user_001",
                    "thread_id": "thread_001",
                    "role": "user",
                    "timestamp": "2026-09-20T10:01:00+00:00",
                },
            )
        ]

    def add(self, **kwargs) -> None:
        pass

    def delete(self, **kwargs) -> None:
        pass


class FakeLLM:
    def invoke(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        return LLMResponse(
            content="Earlier context covered the PDF agent."
        )


def _conversation() -> Conversation:
    return Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            ConversationMessage(
                message_id="msg_1",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="We need to support scanned PDFs.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    10,
                    0,
                    tzinfo=timezone.utc,
                ),
            ),
            ConversationMessage(
                message_id="msg_2",
                user_id="user_001",
                thread_id="thread_001",
                role="assistant",
                content="We discussed the PDF agent.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    10,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
            ConversationMessage(
                message_id="msg_3",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="We also need OCR support.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    10,
                    2,
                    tzinfo=timezone.utc,
                ),
            ),
        ],
    )


def test_stm_pipeline_retrieves_and_expands_context() -> None:
    retriever = STMRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
    )

    manager = STMManager(
        retriever=retriever,
        context_expander=STMContextExpander(
            window_size=1,
            recent_message_count=1,
        ),
        context_budget=STMContextBudget(
            max_tokens=100,
        ),
        summarizer=STMContextSummarizer(
            llm=FakeLLM(),
        ),
    )

    query = STMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="What did we discuss about the PDF agent?",
        top_k=3,
        include_recent_messages=True,
        recency_weight=0.0,
    )

    result = manager.get_context(
        query=query,
        conversation=_conversation(),
    )

    result_ids = [item.item_id for item in result]

    assert result_ids == [
        "msg_1",
        "msg_2",
        "msg_3",
    ]

    assert result[0].content == (
        "We need to support scanned PDFs."
    )
    assert result[1].content == (
        "We discussed the PDF agent."
    )
    assert result[2].content == (
        "We also need OCR support."
    )


def test_stm_pipeline_respects_thread_isolation() -> None:
    retriever = STMRetriever(
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
    )

    query = STMQuery(
        user_id="user_001",
        thread_id="thread_002",
        query="What did we discuss?",
        top_k=3,
    )

    result = retriever.retrieve(query)

    assert result.items == []