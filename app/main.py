from app.config.settings import settings
from app.graph.graph import Phase0Graph
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import JsonlConversationStore


def create_runtime() -> AgentRuntime:
    conversation_store = JsonlConversationStore(settings.conversation_data_path)
    graph = Phase0Graph()
    return AgentRuntime(graph=graph, conversation_store=conversation_store)
