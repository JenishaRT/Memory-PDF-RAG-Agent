from app.contracts.conversation import Conversation
from app.contracts.retrieval import RetrievedItem, STMQuery
from app.memory.stm.context import STMContextExpander
from app.memory.stm.context_budget import STMContextBudget
from app.memory.stm.retriever import STMRetriever
from app.memory.stm.summarizer import STMContextSummarizer


class STMManager:
    def __init__(
        self,
        *,
        retriever: STMRetriever,
        context_expander: STMContextExpander,
        context_budget: STMContextBudget,
        summarizer: STMContextSummarizer,
    ) -> None:
        self.retriever = retriever
        self.context_expander = context_expander
        self.context_budget = context_budget
        self.summarizer = summarizer

    def get_context(
        self,
        *,
        query: STMQuery,
        conversation: Conversation,
    ) -> list[RetrievedItem]:
        if query.user_id != conversation.user_id:
            raise ValueError(
                "Conversation user_id does not match STM query"
            )

        if query.thread_id != conversation.thread_id:
            raise ValueError(
                "Conversation thread_id does not match STM query"
            )
        
        result = self.retriever.retrieve(query)

        expanded_items = self.context_expander.expand(
            items=result.items,
            conversation=conversation,
            include_recent_messages=query.include_recent_messages,
        )

        selected_items = self.context_budget.select(
            expanded_items
        )

        if len(selected_items) == len(expanded_items):
            return selected_items

        selected_ids = {
            item.item_id
            for item in selected_items
        }

        omitted_items = [
            item
            for item in expanded_items
            if item.item_id not in selected_ids
        ]

        summary = self.summarizer.summarize(
            omitted_items
        )

        if summary is None:
            return selected_items

        summary_tokens = self.context_budget.estimate_tokens(
            summary.content
        )

        used_tokens = sum(
            self.context_budget.estimate_tokens(item.content)
            for item in selected_items
        )

        if used_tokens + summary_tokens > self.context_budget.max_tokens:
            return selected_items

        return self._restore_order(
            [summary, *selected_items]
        )

    @staticmethod
    def _restore_order(
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        return sorted(
            items,
            key=lambda item: (
                item.metadata.get("timestamp", ""),
                item.metadata.get("context_type", ""),
            ),
        )