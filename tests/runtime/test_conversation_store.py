from app.contracts.conversation import (
    ConversationMessage,
)
from app.runtime.conversation_store import (
    JsonlConversationStore,
)


def test_jsonl_conversation_store_round_trip(
    tmp_path,
) -> None:
    store = JsonlConversationStore(tmp_path)

    message = ConversationMessage(
        user_id="user_001",
        thread_id="thread_001",
        role="user",
        content="Hello",
    )

    store.append_message(message)

    conversation = store.load(
        user_id="user_001",
        thread_id="thread_001",
    )

    assert conversation.user_id == "user_001"
    assert conversation.thread_id == "thread_001"

    assert len(conversation.messages) == 1

    stored_message = conversation.messages[0]

    assert stored_message.message_id == message.message_id
    assert stored_message.content == "Hello"
    assert stored_message.role == "user"