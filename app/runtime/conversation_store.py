from __future__ import annotations

import json
from pathlib import Path

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.errors import ConversationStoreError


class ConversationStore:
    """Public conversation persistence boundary."""

    def load(self, user_id: str, thread_id: str) -> Conversation:
        raise NotImplementedError

    def append(self, message: ConversationMessage) -> None:
        raise NotImplementedError

    def save(self, conversation: Conversation) -> None:
        raise NotImplementedError


class JsonlConversationStore(ConversationStore):
    def __init__(self, root_path: str | Path = "data/conversations") -> None:
        self.root_path = Path(root_path)

    def _path(self, user_id: str, thread_id: str) -> Path:
        return self.root_path / user_id / f"{thread_id}.jsonl"

    def load(self, user_id: str, thread_id: str) -> Conversation:
        path = self._path(user_id, thread_id)
        if not path.exists():
            return Conversation(user_id=user_id, thread_id=thread_id)

        messages: list[ConversationMessage] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        messages.append(ConversationMessage.model_validate_json(line))
        except (OSError, ValueError) as exc:
            raise ConversationStoreError(
                f"Failed to load conversation {user_id}/{thread_id}: {exc}"
            ) from exc

        return Conversation(
            user_id=user_id,
            thread_id=thread_id,
            messages=messages,
        )

    def append(self, message: ConversationMessage) -> None:
        path = self._path(message.user_id, message.thread_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(message.model_dump_json() + "\n")
        except OSError as exc:
            raise ConversationStoreError(
                f"Failed to append conversation message: {exc}"
            ) from exc

    def save(self, conversation: Conversation) -> None:
        path = self._path(conversation.user_id, conversation.thread_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for message in conversation.messages:
                    handle.write(message.model_dump_json() + "\n")
        except OSError as exc:
            raise ConversationStoreError(
                f"Failed to save conversation: {exc}"
            ) from exc
