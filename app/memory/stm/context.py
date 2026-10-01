from app.contracts.conversation import Conversation
from app.contracts.retrieval import RetrievedItem


class STMContextExpander:
    def __init__(
        self,
        *,
        window_size: int = 1,
        recent_message_count: int = 3,
    ) -> None:
        self.window_size = max(0, window_size)
        self.recent_message_count = max(
            0,
            recent_message_count,
        )

    def expand(
        self,
        *,
        items: list[RetrievedItem],
        conversation: Conversation,
        include_recent_messages: bool = True,
    ) -> list[RetrievedItem]:
        self._validate_conversation(conversation)

        if not items and not include_recent_messages:
            return []

        messages = conversation.messages

        message_indexes = {
            message.message_id: index
            for index, message in enumerate(messages)
        }

        expanded: list[RetrievedItem] = []
        seen_ids: set[str] = set()

        for item in items:
            index = message_indexes.get(item.item_id)

            if index is None:
                continue

            start = max(
                0,
                index - self.window_size,
            )
            end = min(
                len(messages),
                index + self.window_size + 1,
            )

            for message in messages[start:end]:
                if message.message_id in seen_ids:
                    continue

                expanded.append(
                    self._message_to_item(
                        message_id=message.message_id,
                        user_id=conversation.user_id,
                        thread_id=conversation.thread_id,
                        role=message.role,
                        content=message.content,
                        timestamp=message.timestamp.isoformat(),
                        score=(
                            item.score
                            if message.message_id == item.item_id
                            else None
                        ),
                        metadata={
                            "context_type": (
                                "retrieved"
                                if message.message_id == item.item_id
                                else "expanded"
                            ),
                            "expanded_from": item.item_id,
                        },
                    )
                )

                seen_ids.add(message.message_id)

        if include_recent_messages and self.recent_message_count > 0:
            recent_messages = messages[-self.recent_message_count:]

            for message in recent_messages:
                if message.message_id in seen_ids:
                    continue

                expanded.append(
                    self._message_to_item(
                        message_id=message.message_id,
                        user_id=conversation.user_id,
                        thread_id=conversation.thread_id,
                        role=message.role,
                        content=message.content,
                        timestamp=message.timestamp.isoformat(),
                        score=None,
                        metadata={
                            "context_type": "recent",
                        },
                    )
                )

                seen_ids.add(message.message_id)

        return self._sort_by_conversation_order(
            expanded,
            messages,
        )

    @staticmethod
    def _validate_conversation(
        conversation: Conversation,
    ) -> None:
        for message in conversation.messages:
            if message.user_id != conversation.user_id:
                raise ValueError(
                    "Conversation contains a message "
                    "from another user"
                )

            if message.thread_id != conversation.thread_id:
                raise ValueError(
                    "Conversation contains a message "
                    "from another thread"
                )

    @staticmethod
    def _message_to_item(
        *,
        message_id: str,
        user_id: str,
        thread_id: str,
        role: str,
        content: str,
        timestamp: str,
        score: float | None,
        metadata: dict[str, object],
    ) -> RetrievedItem:
        return RetrievedItem(
            item_id=message_id,
            source="stm",
            content=content,
            score=score,
            rank=None,
            metadata={
                "user_id": user_id,
                "thread_id": thread_id,
                "role": role,
                "timestamp": timestamp,
                **metadata,
            },
        )

    @staticmethod
    def _sort_by_conversation_order(
        items: list[RetrievedItem],
        messages: list,
    ) -> list[RetrievedItem]:
        positions = {
            message.message_id: index
            for index, message in enumerate(messages)
        }

        return sorted(
            items,
            key=lambda item: positions.get(
                item.item_id,
                len(messages),
            ),
        )