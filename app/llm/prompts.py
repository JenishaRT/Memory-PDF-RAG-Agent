from langchain_core.prompts import ChatPromptTemplate

from app.contracts.errors import ContractError
from app.llm.provider import ChatMessage


def render_chat_prompt(
    prompt: ChatPromptTemplate,
    **values: str,
) -> list[ChatMessage]:
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }
    messages: list[ChatMessage] = []

    for message in prompt.format_messages(**values):
        role = role_map.get(message.type)
        if role is None:
            raise ContractError(
                f"Unsupported prompt message type: {message.type}"
            )
        if not isinstance(message.content, str):
            raise ContractError(
                "Prompt messages must have text content"
            )
        messages.append(
            ChatMessage(role=role, content=message.content)
        )

    return messages