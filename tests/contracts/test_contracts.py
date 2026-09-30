from app.contracts.conversation import (
    ConversationMessage,
)
from app.contracts.memory import (
    CandidateMemory,
    MemorySource,
)
from app.contracts.routing import RetrievalPlan


def test_conversation_message_contract() -> None:
    message = ConversationMessage(
        user_id="user_001",
        thread_id="thread_001",
        role="user",
        content="Hello",
    )

    assert message.user_id == "user_001"
    assert message.thread_id == "thread_001"
    assert message.role == "user"
    assert message.content == "Hello"
    assert message.message_id


def test_retrieval_plan_supports_multiple_sources() -> None:
    plan = RetrievalPlan(
        use_stm=True,
        use_ltm=True,
        use_pdf=True,
    )

    assert plan.use_stm is True
    assert plan.use_ltm is True
    assert plan.use_pdf is True
    assert plan.any_source_selected is True


def test_retrieval_plan_supports_no_sources() -> None:
    plan = RetrievalPlan()

    assert plan.use_stm is False
    assert plan.use_ltm is False
    assert plan.use_pdf is False
    assert plan.any_source_selected is False


def test_candidate_memory_contract() -> None:
    source = MemorySource(
        thread_id="thread_001",
        message_ids=["message_001"],
    )

    memory = CandidateMemory(
        user_id="user_001",
        memory_type="preference",
        content="The user prefers concise responses.",
        source=source,
        confidence=0.95,
    )

    assert memory.user_id == "user_001"
    assert memory.memory_type == "preference"
    assert memory.confidence == 0.95