import json

from langchain_core.prompts import ChatPromptTemplate

from app.contracts.memory import (
    CandidateMemory,
    MemoryDecision,
    MemoryRecord,
)
from app.llm.prompts import render_chat_prompt
from app.llm.provider import LLMProvider
from app.memory.ltm.schemas import LTMConsolidationOutput


_CONSOLIDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Compare a candidate memory with related existing memories and "
            "choose exactly one action: ADD, UPDATE, MERGE, SUPERSEDE, or "
            "IGNORE. Use only existing memory IDs provided in the input; "
            "never invent IDs. Keep the reason concise and do not include "
            "chain-of-thought.",
        ),
        ("human", "{consolidation_data}"),
    ]
)


class LTMConsolidator:
    def __init__(
        self,
        *,
        llm: LLMProvider,
    ) -> None:
        self.llm = llm

    def consolidate(
        self,
        *,
        candidate: CandidateMemory,
        related_memories: list[MemoryRecord],
    ) -> MemoryDecision:
        if not related_memories:
            return MemoryDecision(
                action="ADD",
                candidate=candidate,
                reason="No related active memory exists.",
            )

        prompt = self._build_prompt(
            candidate=candidate,
            related_memories=related_memories,
        )

        output = self.llm.structured_output(
            render_chat_prompt(
                _CONSOLIDATION_PROMPT,
                consolidation_data=prompt,
            ),
            LTMConsolidationOutput,
            temperature=0.0,
        )

        return self._to_decision(
            output=output,
            candidate=candidate,
            related_memories=related_memories,
        )

    @staticmethod
    def _build_prompt(
        *,
        candidate: CandidateMemory,
        related_memories: list[MemoryRecord],
    ) -> str:
        memories = [
            {
                "memory_id": memory.memory_id,
                "memory_type": memory.memory_type,
                "content": memory.content,
                "version": memory.version,
                "status": memory.status,
            }
            for memory in related_memories
        ]

        return json.dumps(
            {
                "candidate": {
                    "memory_type": candidate.memory_type,
                    "content": candidate.content,
                    "confidence": candidate.confidence,
                },
                "related_memories": memories,
            },
            indent=2,
        )

    @staticmethod
    def _to_decision(
        *,
        output: LTMConsolidationOutput,
        candidate: CandidateMemory,
        related_memories: list[MemoryRecord],
    ) -> MemoryDecision:
        action = output.action
        existing_ids = output.existing_memory_ids

        valid_ids = {
            memory.memory_id
            for memory in related_memories
        }

        if any(
            memory_id not in valid_ids
            for memory_id in existing_ids
        ):
            raise ValueError(
                "Consolidator returned an unknown memory ID"
            )

        if action != "ADD" and not existing_ids:
            raise ValueError(
                "Non-ADD consolidation actions require "
                "existing_memory_ids"
            )

        return MemoryDecision(
            action=action,
            candidate=candidate,
            existing_memory_ids=existing_ids,
            reason=output.reason,
        )