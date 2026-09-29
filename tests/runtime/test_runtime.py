import json

from app.graph.graph import Phase0Graph
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import JsonlConversationStore


def test_runtime_persists_user_and_assistant_messages(tmp_path) -> None:
    store = JsonlConversationStore(tmp_path)
    runtime = AgentRuntime(Phase0Graph(), store)

    response = runtime.run("user_001", "thread_001", "Hello")

    assert response.user_id == "user_001"
    assert response.thread_id == "thread_001"
    assert response.trace_id.startswith("trace_")

    path = tmp_path / "user_001" / "thread_001.jsonl"
    assert path.exists()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["role"] == "user"
    assert records[1]["role"] == "assistant"


def test_runtime_loads_existing_conversation(tmp_path) -> None:
    store = JsonlConversationStore(tmp_path)
    runtime = AgentRuntime(Phase0Graph(), store)

    runtime.run("user_001", "thread_001", "First")
    runtime.run("user_001", "thread_001", "Second")

    conversation = store.load("user_001", "thread_001")
    assert len(conversation.messages) == 4
    assert conversation.messages[0].content == "First"
    assert conversation.messages[2].content == "Second"
