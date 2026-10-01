from datetime import datetime, timezone
from unittest.mock import Mock
import pytest

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.retrieval import RetrievalResult, RetrievedItem, STMQuery
from app.memory.stm.manager import STMManager


def _message(
    message_id: str,
    content: str,
    timestamp: str,
) -> ConversationMessage:
    return ConversationMessage(
        message_id=message_id,
        user_id="user_001",
        thread_id="thread_001",
        role="user",
        content=content,
        timestamp=datetime.fromisoformat(timestamp),
    )


def _item(
    item_id: str,
    content: str,
    timestamp: str,
    context_type: str = "retrieved",
) -> RetrievedItem:
    return RetrievedItem(
        item_id=item_id,
        source="stm",
        content=content,
        score=0.9,
        rank=1,
        metadata={
            "user_id": "user_001",
            "thread_id": "thread_001",
            "role": "user",
            "timestamp": timestamp,
            "context_type": context_type,
        },
    )


def _conversation() -> Conversation:
    return Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            _message(
                "msg_1",
                "First message.",
                "2026-09-20T10:00:00+00:00",
            ),
            _message(
                "msg_2",
                "Second message.",
                "2026-09-20T10:01:00+00:00",
            ),
            _message(
                "msg_3",
                "Third message.",
                "2026-09-20T10:02:00+00:00",
            ),
        ],
    )


def _query() -> STMQuery:
    return STMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="What were we discussing?",
        top_k=3,
    )


def test_manager_returns_selected_context() -> None:
    retriever = Mock()
    expander = Mock()
    budget = Mock()
    summarizer = Mock()

    retrieved_items = [
        _item(
            "msg_1",
            "First message.",
            "2026-09-20T10:00:00+00:00",
        ),
        _item(
            "msg_2",
            "Second message.",
            "2026-09-20T10:01:00+00:00",
        ),
    ]

    retriever.retrieve.return_value = RetrievalResult(
        source="stm",
        query="What were we discussing?",
        items=retrieved_items,
    )

    expander.expand.return_value = retrieved_items
    budget.select.return_value = retrieved_items

    manager = STMManager(
        retriever=retriever,
        context_expander=expander,
        context_budget=budget,
        summarizer=summarizer,
    )

    result = manager.get_context(
        query=_query(),
        conversation=_conversation(),
    )

    assert result == retrieved_items
    summarizer.summarize.assert_not_called()


def test_manager_summarizes_omitted_context() -> None:
    retriever = Mock()
    expander = Mock()
    budget = Mock()
    summarizer = Mock()

    first = _item(
        "msg_1",
        "First message.",
        "2026-09-20T10:00:00+00:00",
    )
    second = _item(
        "msg_2",
        "Second message.",
        "2026-09-20T10:01:00+00:00",
    )
    third = _item(
        "msg_3",
        "Third message.",
        "2026-09-20T10:02:00+00:00",
    )

    expanded_items = [first, second, third]

    retriever.retrieve.return_value = RetrievalResult(
        source="stm",
        query="What were we discussing?",
        items=expanded_items,
    )

    expander.expand.return_value = expanded_items
    budget.select.return_value = [second, third]

    summary = _item(
        "stm_summary",
        "The earlier discussion covered the first message.",
        "2026-09-20T10:00:30+00:00",
        context_type="summary",
    )

    summarizer.summarize.return_value = summary
    budget.estimate_tokens.return_value = 2
    budget.max_tokens = 20

    manager = STMManager(
        retriever=retriever,
        context_expander=expander,
        context_budget=budget,
        summarizer=summarizer,
    )

    result = manager.get_context(
        query=_query(),
        conversation=_conversation(),
    )

    assert summary in result
    assert second in result
    assert third in result

    summarizer.summarize.assert_called_once_with(
        [first]
    )


def test_manager_does_not_add_summary_when_it_exceeds_budget() -> None:
    retriever = Mock()
    expander = Mock()
    budget = Mock()
    summarizer = Mock()

    first = _item(
        "msg_1",
        "First message.",
        "2026-09-20T10:00:00+00:00",
    )
    second = _item(
        "msg_2",
        "Second message.",
        "2026-09-20T10:01:00+00:00",
    )

    expanded_items = [first, second]

    retriever.retrieve.return_value = RetrievalResult(
        source="stm",
        query="What were we discussing?",
        items=expanded_items,
    )

    expander.expand.return_value = expanded_items
    budget.select.return_value = [second]

    summary = _item(
        "stm_summary",
        "Summary of the omitted context.",
        "2026-09-20T10:00:30+00:00",
        context_type="summary",
    )

    summarizer.summarize.return_value = summary
    budget.max_tokens = 5
    budget.estimate_tokens.side_effect = [
        4,
        4,
    ]

    manager = STMManager(
        retriever=retriever,
        context_expander=expander,
        context_budget=budget,
        summarizer=summarizer,
    )

    result = manager.get_context(
        query=_query(),
        conversation=_conversation(),
    )

    assert result == [second]


def test_manager_passes_recent_message_setting_to_expander() -> None:
    retriever = Mock()
    expander = Mock()
    budget = Mock()
    summarizer = Mock()

    items = [
        _item(
            "msg_1",
            "First message.",
            "2026-09-20T10:00:00+00:00",
        )
    ]

    retriever.retrieve.return_value = RetrievalResult(
        source="stm",
        query="What happened?",
        items=items,
    )

    expander.expand.return_value = items
    budget.select.return_value = items

    query = STMQuery(
        user_id="user_001",
        thread_id="thread_001",
        query="What happened?",
        top_k=3,
        include_recent_messages=False,
    )

    conversation = _conversation()

    manager = STMManager(
        retriever=retriever,
        context_expander=expander,
        context_budget=budget,
        summarizer=summarizer,
    )

    manager.get_context(
        query=query,
        conversation=conversation,
    )

    expander.expand.assert_called_once_with(
        items=items,
        conversation=conversation,
        include_recent_messages=False,
    )

@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user_id", "user_002", "user_id"),
        ("thread_id", "thread_002", "thread_id"),
    ],
)
def test_manager_rejects_mismatched_conversation(
    field: str,
    value: str,
    message: str,
) -> None:
    retriever = Mock()
    manager = STMManager(
        retriever=retriever,
        context_expander=Mock(),
        context_budget=Mock(),
        summarizer=Mock(),
    )
    conversation = _conversation().model_copy(
        update={field: value}
    )

    with pytest.raises(ValueError, match=message):
        manager.get_context(
            query=_query(),
            conversation=conversation,
        )

    retriever.retrieve.assert_not_called()