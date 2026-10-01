from app.contracts.retrieval import RetrievedItem
from app.memory.stm.context_budget import STMContextBudget


def test_context_budget_selects_items_within_limit() -> None:
    budget = STMContextBudget(
        max_tokens=20,
        chars_per_token=4,
    )

    items = [
        RetrievedItem(
            item_id="message_001",
            source="stm",
            content="A" * 40,
            score=0.9,
            rank=1,
            metadata={
                "context_type": "retrieved",
                "timestamp": "2026-09-20T10:00:00+00:00",
            },
        ),
        RetrievedItem(
            item_id="message_002",
            source="stm",
            content="B" * 40,
            score=0.8,
            rank=2,
            metadata={
                "context_type": "expanded",
                "timestamp": "2026-09-20T10:01:00+00:00",
            },
        ),
        RetrievedItem(
            item_id="message_003",
            source="stm",
            content="C" * 40,
            score=0.7,
            rank=3,
            metadata={
                "context_type": "recent",
                "timestamp": "2026-09-20T10:02:00+00:00",
            },
        ),
    ]

    result = budget.select(items)

    assert len(result) == 2
    assert sum(
        budget.estimate_tokens(item.content)
        for item in result
    ) <= 20


def test_context_budget_prioritizes_retrieved_messages() -> None:
    budget = STMContextBudget(
        max_tokens=5,
        chars_per_token=4,
    )

    items = [
        RetrievedItem(
            item_id="expanded",
            source="stm",
            content="E" * 20,
            score=1.0,
            rank=1,
            metadata={
                "context_type": "expanded",
                "timestamp": "2026-09-20T10:00:00+00:00",
            },
        ),
        RetrievedItem(
            item_id="retrieved",
            source="stm",
            content="R" * 20,
            score=0.5,
            rank=2,
            metadata={
                "context_type": "retrieved",
                "timestamp": "2026-09-20T10:01:00+00:00",
            },
        ),
    ]

    result = budget.select(items)

    assert len(result) == 1
    assert result[0].item_id == "retrieved"


def test_context_budget_uses_score_for_same_context_type() -> None:
    budget = STMContextBudget(
        max_tokens=5,
        chars_per_token=4,
    )

    items = [
        RetrievedItem(
            item_id="low_score",
            source="stm",
            content="L" * 20,
            score=0.2,
            rank=1,
            metadata={
                "context_type": "retrieved",
                "timestamp": "2026-09-20T10:00:00+00:00",
            },
        ),
        RetrievedItem(
            item_id="high_score",
            source="stm",
            content="H" * 20,
            score=0.9,
            rank=2,
            metadata={
                "context_type": "retrieved",
                "timestamp": "2026-09-20T10:01:00+00:00",
            },
        ),
    ]

    result = budget.select(items)

    assert len(result) == 1
    assert result[0].item_id == "high_score"


def test_context_budget_restores_conversation_order() -> None:
    budget = STMContextBudget(
        max_tokens=20,
        chars_per_token=4,
    )

    items = [
        RetrievedItem(
            item_id="message_002",
            source="stm",
            content="Second message.",
            score=0.9,
            rank=1,
            metadata={
                "context_type": "retrieved",
                "timestamp": "2026-09-20T10:01:00+00:00",
            },
        ),
        RetrievedItem(
            item_id="message_001",
            source="stm",
            content="First message.",
            score=0.5,
            rank=2,
            metadata={
                "context_type": "expanded",
                "timestamp": "2026-09-20T10:00:00+00:00",
            },
        ),
    ]

    result = budget.select(items)

    assert [
        item.item_id
        for item in result
    ] == [
        "message_001",
        "message_002",
    ]


def test_context_budget_handles_empty_items() -> None:
    budget = STMContextBudget(max_tokens=100)

    assert budget.select([]) == []


def test_context_budget_rejects_invalid_budget() -> None:
    try:
        STMContextBudget(max_tokens=0)
    except ValueError as exc:
        assert "max_tokens" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid token budget to fail"
        )


def test_context_budget_estimates_tokens() -> None:
    budget = STMContextBudget(
        max_tokens=100,
        chars_per_token=4,
    )

    assert budget.estimate_tokens("") == 0
    assert budget.estimate_tokens("1234") == 1
    assert budget.estimate_tokens("12345678") == 2