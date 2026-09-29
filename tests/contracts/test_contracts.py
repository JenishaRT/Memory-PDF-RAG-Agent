from app.contracts import (
    AgentRequest,
    AgentResponse,
    Conversation,
    DocumentChunk,
    MemoryRecord,
    RetrievalPlan,
    RetrievedItem,
)


def test_retrieval_plan_supports_multiple_sources() -> None:
    plan = RetrievalPlan(use_stm=True, use_ltm=True, use_pdf=True)
    assert plan.use_stm and plan.use_ltm and plan.use_pdf


def test_retrieved_item_preserves_source() -> None:
    item = RetrievedItem(source="pdf", id="chunk_1", content="evidence")
    assert item.source == "pdf"


def test_core_models_validate() -> None:
    request = AgentRequest(
        user_id="user_001",
        thread_id="thread_001",
        message="hello",
    )
    response = AgentResponse(
        user_id=request.user_id,
        thread_id=request.thread_id,
        answer="hello",
        trace_id="trace_1",
    )
    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_001",
    )
    assert response.thread_id == conversation.thread_id


def test_document_chunk_preserves_provenance() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        filename="example.pdf",
        page_number=2,
        section="Introduction",
        content="text",
    )
    assert chunk.page_number == 2


def test_memory_record_defaults_to_active() -> None:
    from datetime import datetime, timezone

    memory = MemoryRecord(
        memory_id="mem_1",
        user_id="user_001",
        memory_type="preference",
        content="Prefers concise answers",
        importance=0.8,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    assert memory.status.value == "active"
