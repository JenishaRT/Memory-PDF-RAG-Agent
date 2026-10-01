from unittest.mock import Mock

from app.contracts.retrieval import RetrievedItem
from app.llm.provider import LLMResponse
from app.memory.stm.summarizer import STMContextSummarizer
from app.llm.provider import LLMResponse


def _item(
    item_id: str,
    content: str,
    timestamp: str,
    role: str = "user",
) -> RetrievedItem:
    return RetrievedItem(
        item_id=item_id,
        source="stm",
        content=content,
        score=0.8,
        rank=1,
        metadata={
            "role": role,
            "timestamp": timestamp,
            "context_type": "expanded",
        },
    )


def test_summarizer_returns_summary() -> None:
    llm = Mock()
    llm.invoke.return_value = LLMResponse(
        content=(
            "The user is building a memory system "
            "with STM and LTM."
        )
    )

    summarizer = STMContextSummarizer(llm=llm)

    result = summarizer.summarize(
        [
            _item(
                "msg_1",
                "I am building a memory system.",
                "2026-09-20T10:00:00+00:00",
            ),
            _item(
                "msg_2",
                "It will have STM and LTM.",
                "2026-09-20T10:01:00+00:00",
            ),
        ]
    )

    assert result is not None
    assert result.item_id == "stm_summary"
    assert result.source == "stm"
    assert (
        result.content
        == "The user is building a memory system "
        "with STM and LTM."
    )
    assert result.metadata["context_type"] == "summary"
    assert result.metadata["source_message_ids"] == [
        "msg_1",
        "msg_2",
    ]


def test_summarizer_sends_messages_in_chronological_order() -> None:
    llm = Mock()
    llm.invoke.return_value = LLMResponse(content="Summary")

    summarizer = STMContextSummarizer(llm=llm)

    summarizer.summarize(
        [
            _item(
                "msg_2",
                "Second message.",
                "2026-09-20T10:01:00+00:00",
            ),
            _item(
                "msg_1",
                "First message.",
                "2026-09-20T10:00:00+00:00",
            ),
        ]
    )

    messages = llm.invoke.call_args.args[0]

    assert "First message." in messages[1].content
    assert "Second message." in messages[1].content

    first_position = messages[1].content.index(
        "First message."
    )
    second_position = messages[1].content.index(
        "Second message."
    )

    assert first_position < second_position


def test_summarizer_returns_none_for_empty_items() -> None:
    llm = Mock()

    summarizer = STMContextSummarizer(llm=llm)

    result = summarizer.summarize([])

    assert result is None
    llm.invoke.assert_not_called()


def test_summarizer_returns_none_for_empty_llm_response() -> None:
    llm = Mock()
    llm.invoke.return_value = LLMResponse(content="   ")

    summarizer = STMContextSummarizer(llm=llm)

    result = summarizer.summarize(
        [
            _item(
                "msg_1",
                "Some context.",
                "2026-09-20T10:00:00+00:00",
            )
        ]
    )

    assert result is None


def test_summarizer_preserves_source_message_ids() -> None:
    llm = Mock()
    llm.invoke.return_value = LLMResponse(content="Summary")

    summarizer = STMContextSummarizer(llm=llm)

    result = summarizer.summarize(
        [
            _item(
                "msg_1",
                "First.",
                "2026-09-20T10:00:00+00:00",
            ),
            _item(
                "msg_2",
                "Second.",
                "2026-09-20T10:01:00+00:00",
            ),
        ]
    )

    assert result is not None
    assert result.metadata["source_message_ids"] == [
        "msg_1",
        "msg_2",
    ]