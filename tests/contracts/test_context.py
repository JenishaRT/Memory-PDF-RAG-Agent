from app.contracts.context import (
    AgentContext,
    ContextItem,
)


def test_context_can_store_items() -> None:
    context = AgentContext(
        max_tokens=6000,
    )

    context.add(
        ContextItem(
            source="stm",
            item_id="message_1",
            content="Hello",
            score=0.9,
        )
    )

    assert not context.is_empty
    assert len(context.items) == 1
    assert context.items[0].source == "stm"


def test_empty_context() -> None:
    context = AgentContext()

    assert context.is_empty
    assert context.items == []