from app.contracts.runtime import AgentRequest
from app.graph.graph import Phase0Graph
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import (
    JsonlConversationStore,
)


def test_phase0_runtime_flow(tmp_path) -> None:
    store = JsonlConversationStore(tmp_path)
    graph = Phase0Graph()

    runtime = AgentRuntime(
        conversation_store=store,
        graph=graph,
    )

    request = AgentRequest(
        user_id="user_001",
        thread_id="thread_001",
        message="Hello",
    )

    response = runtime.handle(request)

    assert response.user_id == "user_001"
    assert response.thread_id == "thread_001"
    assert response.trace_id
    assert "Phase 0 skeleton" in response.answer

    conversation = store.load(
        "user_001",
        "thread_001",
    )

    assert len(conversation.messages) == 2

    user_message = conversation.messages[0]
    assistant_message = conversation.messages[1]

    assert user_message.role == "user"
    assert user_message.content == "Hello"

    assert assistant_message.role == "assistant"
    assert assistant_message.content == response.answer