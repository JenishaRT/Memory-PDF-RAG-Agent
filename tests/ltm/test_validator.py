import pytest

from app.contracts.memory import CandidateMemory, MemorySource
from app.memory.ltm.validator import LTMValidator


def _candidate(
    **overrides,
) -> CandidateMemory:
    values = {
        "user_id": "user_001",
        "memory_type": "preference",
        "content": "User prefers concise responses.",
        "source": MemorySource(
            thread_id="thread_001",
            message_ids=["message_001"],
        ),
        "confidence": 0.9,
    }

    values.update(overrides)

    return CandidateMemory(**values)


def test_validator_accepts_valid_candidate() -> None:
    validator = LTMValidator()

    validator.validate(_candidate())


def test_validator_rejects_empty_user_id() -> None:
    validator = LTMValidator()

    with pytest.raises(
        ValueError,
        match="must have a user_id",
    ):
        validator.validate(
            _candidate(user_id=" ")
        )


def test_validator_rejects_empty_content() -> None:
    validator = LTMValidator()

    with pytest.raises(
        ValueError,
        match="content cannot be empty",
    ):
        validator.validate(
            _candidate(content=" ")
        )


def test_validator_rejects_content_that_is_too_long() -> None:
    validator = LTMValidator(
        max_content_length=20
    )

    with pytest.raises(
        ValueError,
        match="content is too long",
    ):
        validator.validate(
            _candidate(
                content="This memory is definitely longer than twenty characters."
            )
        )


def test_validator_rejects_empty_source_thread() -> None:
    validator = LTMValidator()

    source = MemorySource(
        thread_id=" ",
        message_ids=["message_001"],
    )

    with pytest.raises(
        ValueError,
        match="source thread_id",
    ):
        validator.validate(
            _candidate(source=source)
        )


def test_validator_rejects_missing_source_messages() -> None:
    validator = LTMValidator()

    source = MemorySource(
        thread_id="thread_001",
        message_ids=[],
    )

    with pytest.raises(
        ValueError,
        match="source message_ids",
    ):
        validator.validate(
            _candidate(source=source)
        )


def test_validator_rejects_empty_source_message_id() -> None:
    validator = LTMValidator()

    source = MemorySource(
        thread_id="thread_001",
        message_ids=["message_001", " "],
    )

    with pytest.raises(
        ValueError,
        match="empty message_id",
    ):
        validator.validate(
            _candidate(source=source)
        )


def test_candidate_confidence_is_validated_by_contract() -> None:
    with pytest.raises(ValueError):
        _candidate(confidence=1.5)


def test_validator_does_not_modify_candidate() -> None:
    validator = LTMValidator()
    candidate = _candidate()

    original_content = candidate.content

    validator.validate(candidate)

    assert candidate.content == original_content