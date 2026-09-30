from dataclasses import dataclass

from app.contracts.conversation import Conversation


@dataclass
class AgentSession:
    user_id: str
    thread_id: str
    conversation: Conversation