from pathlib import Path
import json

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.memory import (
    CandidateMemory,
    MemoryDecision,
    MemorySource,
)
from app.contracts.retrieval import LTMQuery, RetrievedItem
from app.memory.ltm.extractor import LTMExtractor
from app.memory.ltm.manager import LTMManager
from app.memory.ltm.related import LTMRelatedMemoryRetriever
from app.memory.ltm.retriever import LTMRetriever
from app.memory.ltm.store import JsonMemoryStore
from app.memory.ltm.validator import LTMValidator
from app.memory.ltm.writer import LTMWriter


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def structured_output(
        self,
        messages,
        output_schema,
        *,
        temperature: float | None = None,
    ):
        return output_schema.model_validate(json.loads(self.response))


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self):
        self.items = {}

    def add(
        self,
        *,
        collection,
        ids,
        documents,
        embeddings,
        metadatas,
    ):
        for item_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            self.items[item_id] = RetrievedItem(
                item_id=item_id,
                source="ltm",
                content=document,
                score=1.0,
                rank=1,
                metadata=metadata,
            )

    def search(
        self,
        *,
        collection,
        query_embedding,
        top_k,
        filters=None,
    ):
        items = list(self.items.values())

        if filters:
            items = [
                item
                for item in items
                if all(
                    item.metadata.get(key) == value
                    for key, value in filters.items()
                )
            ]

        return items[:top_k]

    def delete(self, *, collection, ids):
        for item_id in ids:
            self.items.pop(item_id, None)


class FakeConsolidator:
    def consolidate(self, *, candidate, related_memories):
        if related_memories:
            return MemoryDecision(
                action="IGNORE",
                candidate=candidate,
                existing_memory_ids=[
                    related_memories[0].memory_id
                ],
                reason="Duplicate memory.",
            )

        return MemoryDecision(
            action="ADD",
            candidate=candidate,
            reason="New memory.",
        )


def test_ltm_pipeline_extracts_persists_and_retrieves_memory(
    tmp_path: Path,
):
    user_id = "user_001"
    thread_id = "thread_001"

    message = ConversationMessage(
        user_id=user_id,
        thread_id=thread_id,
        role="user",
        content="I prefer Chroma for the vector database.",
    )

    conversation = Conversation(
        user_id=user_id,
        thread_id=thread_id,
        messages=[message],
    )

    extractor_response = """
    {
        "memories": [
            {
                "memory_type": "preference",
                "content": "User prefers Chroma for the vector database.",
                "confidence": 0.95,
                "message_ids": ["MESSAGE_ID"]
            }
        ]
    }
    """.replace("MESSAGE_ID", message.message_id)

    extractor = LTMExtractor(
        llm=FakeLLM(extractor_response),
    )

    candidates = extractor.extract(conversation)

    assert len(candidates) == 1

    candidate = candidates[0]

    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddings()

    memory_store = JsonMemoryStore(
        path=tmp_path / "memories.json",
    )

    writer = LTMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    related_retriever = LTMRelatedMemoryRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        memory_store=memory_store,
    )

    manager = LTMManager(
        memory_store=memory_store,
        validator=LTMValidator(),
        related_retriever=related_retriever,
        consolidator=FakeConsolidator(),
        writer=writer,
    )

    decision = manager.process(candidate)

    assert decision.action == "ADD"

    memories = memory_store.list(
        user_id=user_id,
        include_inactive=False,
    )

    assert len(memories) == 1
    assert memories[0].content == (
        "User prefers Chroma for the vector database."
    )
    assert memories[0].status == "active"

    query = LTMQuery(
        user_id=user_id,
        thread_id=thread_id,
        query="What vector database does the user prefer?",
        top_k=3,
    )

    retriever = LTMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = retriever.retrieve(query)

    assert result.source == "ltm"
    assert len(result.items) == 1
    assert result.items[0].content == (
        "User prefers Chroma for the vector database."
    )
    assert result.items[0].metadata["user_id"] == user_id
    assert result.items[0].metadata["status"] == "active"


class SupersedingConsolidator:
    def consolidate(self, *, candidate, related_memories):
        if related_memories:
            return MemoryDecision(
                action="SUPERSEDE",
                candidate=candidate,
                existing_memory_ids=[
                    related_memories[0].memory_id
                ],
                reason="The user changed their preference.",
            )

        return MemoryDecision(
            action="ADD",
            candidate=candidate,
            reason="New memory.",
        )


def test_ltm_pipeline_supersedes_previous_memory(tmp_path: Path):
    user_id = "user_001"
    first_thread = "thread_001"
    second_thread = "thread_002"

    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddings()

    memory_store = JsonMemoryStore(
        path=tmp_path / "memories.json",
    )

    writer = LTMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    related_retriever = LTMRelatedMemoryRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        memory_store=memory_store,
    )

    manager = LTMManager(
        memory_store=memory_store,
        validator=LTMValidator(),
        related_retriever=related_retriever,
        consolidator=SupersedingConsolidator(),
        writer=writer,
    )

    first_candidate = CandidateMemory(
        user_id=user_id,
        memory_type="preference",
        content="User prefers Chroma for the vector database.",
        confidence=0.95,
        source=MemorySource(
            thread_id=first_thread,
            message_ids=["message_001"],
        ),
    )

    first_decision = manager.process(first_candidate)

    assert first_decision.action == "ADD"

    first_memories = memory_store.list(
        user_id=user_id,
        include_inactive=False,
    )

    assert len(first_memories) == 1

    old_memory = first_memories[0]

    second_candidate = CandidateMemory(
        user_id=user_id,
        memory_type="preference",
        content="User now prefers Qdrant for the vector database.",
        confidence=0.96,
        source=MemorySource(
            thread_id=second_thread,
            message_ids=["message_002"],
        ),
    )

    second_decision = manager.process(second_candidate)

    assert second_decision.action == "SUPERSEDE"
    assert second_decision.existing_memory_ids == [
        old_memory.memory_id
    ]

    all_memories = memory_store.list(
        user_id=user_id,
        include_inactive=True,
    )

    assert len(all_memories) == 2

    stored_old = next(
        memory
        for memory in all_memories
        if memory.memory_id == old_memory.memory_id
    )

    new_memory = next(
        memory
        for memory in all_memories
        if memory.memory_id != old_memory.memory_id
    )

    assert stored_old.status == "superseded"
    assert new_memory.status == "active"
    assert new_memory.content == (
        "User now prefers Qdrant for the vector database."
    )
    assert new_memory.version == old_memory.version + 1
    assert new_memory.supersedes_memory_id == old_memory.memory_id

    query = LTMQuery(
        user_id=user_id,
        thread_id=second_thread,
        query="What vector database does the user prefer?",
        top_k=5,
    )

    retriever = LTMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = retriever.retrieve(query)

    assert len(result.items) == 1
    assert result.items[0].item_id == new_memory.memory_id
    assert result.items[0].content == (
        "User now prefers Qdrant for the vector database."
    )

def test_ltm_retrieval_is_isolated_by_user(tmp_path: Path):
    user_1 = "user_001"
    user_2 = "user_002"
    thread_1 = "thread_001"
    thread_2 = "thread_002"

    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddings()

    memory_store = JsonMemoryStore(
        path=tmp_path / "memories.json",
    )

    writer = LTMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    related_retriever = LTMRelatedMemoryRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        memory_store=memory_store,
    )

    manager = LTMManager(
        memory_store=memory_store,
        validator=LTMValidator(),
        related_retriever=related_retriever,
        consolidator=FakeConsolidator(),
        writer=writer,
    )

    user_1_candidate = CandidateMemory(
        user_id=user_1,
        memory_type="preference",
        content="User prefers Chroma for vector storage.",
        confidence=0.95,
        source=MemorySource(
            thread_id=thread_1,
            message_ids=["message_001"],
        ),
    )

    user_2_candidate = CandidateMemory(
        user_id=user_2,
        memory_type="preference",
        content="User prefers Qdrant for vector storage.",
        confidence=0.95,
        source=MemorySource(
            thread_id=thread_2,
            message_ids=["message_002"],
        ),
    )

    manager.process(user_1_candidate)
    manager.process(user_2_candidate)

    retriever = LTMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    query = LTMQuery(
        user_id=user_1,
        thread_id=thread_1,
        query="Which vector database does the user prefer?",
        top_k=5,
    )

    result = retriever.retrieve(query)

    assert len(result.items) == 1
    assert result.items[0].content == (
        "User prefers Chroma for vector storage."
    )
    assert result.items[0].metadata["user_id"] == user_1

def test_ltm_consolidation_does_not_use_another_users_memory(
    tmp_path: Path,
):
    user_1 = "user_001"
    user_2 = "user_002"

    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddings()

    memory_store = JsonMemoryStore(
        path=tmp_path / "memories.json",
    )

    writer = LTMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    related_retriever = LTMRelatedMemoryRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        memory_store=memory_store,
    )

    manager = LTMManager(
        memory_store=memory_store,
        validator=LTMValidator(),
        related_retriever=related_retriever,
        consolidator=FakeConsolidator(),
        writer=writer,
    )

    other_user_memory = CandidateMemory(
        user_id=user_2,
        memory_type="preference",
        content="User prefers PostgreSQL.",
        confidence=0.95,
        source=MemorySource(
            thread_id="thread_002",
            message_ids=["message_002"],
        ),
    )

    manager.process(other_user_memory)

    candidate = CandidateMemory(
        user_id=user_1,
        memory_type="preference",
        content="User prefers PostgreSQL.",
        confidence=0.95,
        source=MemorySource(
            thread_id="thread_001",
            message_ids=["message_001"],
        ),
    )

    related = related_retriever.retrieve(candidate)

    assert related == []

def test_ltm_full_memory_lifecycle(tmp_path: Path):
    user_id = "user_001"

    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddings()

    memory_store = JsonMemoryStore(
        path=tmp_path / "memories.json",
    )

    writer = LTMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    related_retriever = LTMRelatedMemoryRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        memory_store=memory_store,
    )

    manager = LTMManager(
        memory_store=memory_store,
        validator=LTMValidator(),
        related_retriever=related_retriever,
        consolidator=FakeConsolidator(),
        writer=writer,
    )

    candidate = CandidateMemory(
        user_id=user_id,
        memory_type="preference",
        content="User prefers Chroma.",
        confidence=0.95,
        source=MemorySource(
            thread_id="thread_001",
            message_ids=["message_001"],
        ),
    )

    decision = manager.process(candidate)

    assert decision.action == "ADD"

    memories = memory_store.list(user_id=user_id)
    assert len(memories) == 1
    assert memories[0].status == "active"

    retrieved = LTMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    ).retrieve(
        LTMQuery(
            user_id=user_id,
            thread_id="thread_001",
            query="What database does the user prefer?",
            top_k=5,
        )
    )

    assert len(retrieved.items) == 1
    assert retrieved.items[0].content == "User prefers Chroma."