from datetime import datetime, timezone

from app.memory.stm.models import STMMessage


def test_stm_message() -> None:
    message = STMMessage(
        message_id="message_001",
        user_id="user_001",
        thread_id="thread_001",
        role="user",
        content="I prefer Python.",
        timestamp=datetime.now(timezone.utc),
    )

    assert message.message_id == "message_001"
    assert message.user_id == "user_001"
    assert message.thread_id == "thread_001"
    assert message.role == "user"
    assert message.content == "I prefer Python."