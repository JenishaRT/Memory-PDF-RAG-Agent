from datetime import datetime, timezone

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.retrieval import RetrievedItem
from app.memory.stm.context import STMContextExpander


def _conversation() -> Conversation:
    return Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            ConversationMessage(
                message_id="message_001",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="I am building a PDF agent.",
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
                message_id="message_002",
                user_id="user_001",
                thread_id="thread_001",
                role="assistant",
                content="You can use vector retrieval.",
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
                message_id="message_003",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="I also want cross-document queries.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    10,
                    2,
                    tzinfo=timezone.utc,
                ),
            ),
            ConversationMessage(
                message_id="message_004",
                user_id="user_001",
                thread_id="thread_001",
                role="assistant",
                content="Then retrieve from all documents.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    10,
                    3,
                    tzinfo=timezone.utc,
                ),
            ),
        ],
    )


def test_context_expansion_includes_neighboring_messages() -> None:
    conversation = _conversation()

    items = [
        RetrievedItem(
            item_id="message_002",
            source="stm",
            content="You can use vector retrieval.",
            score=0.9,
            rank=1,
            metadata={},
        )
    ]

    expander = STMContextExpander(
        window_size=1,
        recent_message_count=0,
    )

    result = expander.expand(
        items=items,
        conversation=conversation,
        include_recent_messages=False,
    )

    assert [
        item.item_id
        for item in result
    ] == [
        "message_001",
        "message_002",
        "message_003",
    ]

    assert result[1].metadata["context_type"] == "retrieved"
    assert result[0].metadata["context_type"] == "expanded"
    assert result[2].metadata["context_type"] == "expanded"


def test_context_expansion_includes_recent_messages() -> None:
    conversation = _conversation()

    expander = STMContextExpander(
        window_size=0,
        recent_message_count=2,
    )

    result = expander.expand(
        items=[],
        conversation=conversation,
        include_recent_messages=True,
    )

    assert [
        item.item_id
        for item in result
    ] == [
        "message_003",
        "message_004",
    ]

    assert all(
        item.metadata["context_type"] == "recent"
        for item in result
    )


def test_context_expansion_deduplicates_messages() -> None:
    conversation = _conversation()

    items = [
        RetrievedItem(
            item_id="message_002",
            source="stm",
            content="You can use vector retrieval.",
            score=0.9,
            rank=1,
            metadata={},
        )
    ]

    expander = STMContextExpander(
        window_size=1,
        recent_message_count=3,
    )

    result = expander.expand(
        items=items,
        conversation=conversation,
        include_recent_messages=True,
    )

    ids = [
        item.item_id
        for item in result
    ]

    assert len(ids) == len(set(ids))


def test_context_expansion_prevents_cross_user_messages() -> None:
    conversation = _conversation()

    conversation.messages[2].user_id = "user_002"

    expander = STMContextExpander()

    try:
        expander.expand(
            items=[],
            conversation=conversation,
        )
    except ValueError as exc:
        assert "another user" in str(exc)
    else:
        raise AssertionError(
            "Expected cross-user conversation message "
            "to be rejected"
        )


def test_context_expansion_prevents_cross_thread_messages() -> None:
    conversation = _conversation()

    conversation.messages[2].thread_id = "thread_002"

    expander = STMContextExpander()

    try:
        expander.expand(
            items=[],
            conversation=conversation,
        )
    except ValueError as exc:
        assert "another thread" in str(exc)
    else:
        raise AssertionError(
            "Expected cross-thread conversation message "
            "to be rejected"
        )


def test_context_expansion_ignores_unknown_retrieved_message() -> None:
    conversation = _conversation()

    items = [
        RetrievedItem(
            item_id="unknown_message",
            source="stm",
            content="Unknown message.",
            score=0.9,
            rank=1,
            metadata={},
        )
    ]

    expander = STMContextExpander(
        window_size=1,
        recent_message_count=0,
    )

    result = expander.expand(
        items=items,
        conversation=conversation,
        include_recent_messages=False,
    )

    assert result == []

def test_zero_recent_message_count_adds_no_recent_messages() -> None:
    result = STMContextExpander(
        window_size=0,
        recent_message_count=0,
    ).expand(
        items=[],
        conversation=_conversation(),
        include_recent_messages=True,
    )

    assert result == []