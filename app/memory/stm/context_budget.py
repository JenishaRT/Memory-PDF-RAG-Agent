from app.contracts.retrieval import RetrievedItem


class STMContextBudget:
    def __init__(
        self,
        *,
        max_tokens: int,
        chars_per_token: float = 4.0,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero"
            )

        if chars_per_token <= 0:
            raise ValueError(
                "chars_per_token must be greater than zero"
            )

        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def select(
        self,
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        if not items:
            return []

        selected: list[RetrievedItem] = []
        used_tokens = 0

        for item in self._ordered_items(items):
            item_tokens = self.estimate_tokens(
                item.content
            )

            if item_tokens == 0:
                continue

            if used_tokens + item_tokens > self.max_tokens:
                continue

            selected.append(item)
            used_tokens += item_tokens

        return self._restore_conversation_order(selected)

    def estimate_tokens(self, content: str) -> int:
        if not content:
            return 0

        return max(
            1,
            round(
                len(content) / self.chars_per_token
            ),
        )

    def _ordered_items(
        self,
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        return sorted(
            items,
            key=self._priority,
            reverse=True,
        )

    @staticmethod
    def _priority(
        item: RetrievedItem,
    ) -> tuple[int, float]:
        context_type = item.metadata.get(
            "context_type"
        )

        if context_type == "retrieved":
            priority = 3
        elif context_type == "recent":
            priority = 2
        else:
            priority = 1

        score = item.score or 0.0

        return priority, score

    @staticmethod
    def _restore_conversation_order(
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        return sorted(
            items,
            key=lambda item: (
                item.metadata.get("timestamp", ""),
                item.rank or 0,
            ),
        )