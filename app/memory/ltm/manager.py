from datetime import datetime, timezone

from app.contracts.memory import (
    CandidateMemory,
    MemoryDecision,
    MemoryRecord,
)
from app.memory.ltm.related import LTMRelatedMemoryRetriever
from app.memory.ltm.store import MemoryStore
from app.memory.ltm.validator import LTMValidator
from app.memory.ltm.writer import LTMWriter


class LTMManager:
    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        validator: LTMValidator,
        related_retriever: LTMRelatedMemoryRetriever,
        consolidator,
        writer: LTMWriter,
    ) -> None:
        self.memory_store = memory_store
        self.validator = validator
        self.related_retriever = related_retriever
        self.consolidator = consolidator
        self.writer = writer

    def process(
        self,
        candidate: CandidateMemory,
    ) -> MemoryDecision:
        self.validator.validate(candidate)

        related_memories = (
            self.related_retriever.retrieve(
                candidate
            )
        )

        decision = self.consolidator.consolidate(
            candidate=candidate,
            related_memories=related_memories,
        )

        self._apply_decision(
            decision=decision,
            related_memories=related_memories,
        )

        return decision

    def _apply_decision(
        self,
        *,
        decision: MemoryDecision,
        related_memories: list[MemoryRecord],
    ) -> None:
        if decision.action == "IGNORE":
            return

        if decision.action == "ADD":
            memory = self._create_memory(
                decision.candidate
            )
            self._save_and_index(memory)
            return

        existing = self._get_existing_memories(
            decision,
            related_memories,
        )

        if decision.action == "UPDATE":
            memory = self._update_memory(
                decision.candidate,
                existing[0],
            )
            self._save_and_index(memory)
            return

        if decision.action == "MERGE":
            memory = self._merge_memories(
                decision.candidate,
                existing,
            )

            for existing_memory in existing:
                self._deactivate(
                    existing_memory
                )

            self._save_and_index(memory)
            return

        if decision.action == "SUPERSEDE":
            memory = self._create_memory(
                decision.candidate,
                version=existing[0].version + 1,
                supersedes_memory_id=(
                    existing[0].memory_id
                ),
            )

            self._deactivate(
                existing[0]
            )
            self._save_and_index(memory)
            return

        raise ValueError(
            f"Unsupported memory action: {decision.action}"
        )

    def _save_and_index(
        self,
        memory: MemoryRecord,
    ) -> None:
        self.memory_store.save(memory)
        self.writer.write(memory)

    def _deactivate(
        self,
        memory: MemoryRecord,
    ) -> None:
        inactive = memory.model_copy(
            update={
                "status": "superseded",
                "updated_at": datetime.now(
                    timezone.utc
                ),
            }
        )

        self.memory_store.save(inactive)
        self.writer.remove(
            memory.memory_id
        )

    @staticmethod
    def _get_existing_memories(
        decision: MemoryDecision,
        related_memories: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        memories_by_id = {
            memory.memory_id: memory
            for memory in related_memories
        }

        existing = []

        for memory_id in decision.existing_memory_ids:
            memory = memories_by_id.get(memory_id)

            if memory is None:
                raise ValueError(
                    f"Memory not found: {memory_id}"
                )

            existing.append(memory)

        if not existing:
            raise ValueError(
                "Consolidation action requires an existing memory"
            )

        return existing

    @staticmethod
    def _create_memory(
        candidate: CandidateMemory,
        *,
        version: int = 1,
        supersedes_memory_id: str | None = None,
    ) -> MemoryRecord:
        return MemoryRecord(
            user_id=candidate.user_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            status="active",
            source=candidate.source,
            version=version,
            supersedes_memory_id=(
                supersedes_memory_id
            ),
            metadata=candidate.metadata,
        )

    @staticmethod
    def _update_memory(
        candidate: CandidateMemory,
        existing: MemoryRecord,
    ) -> MemoryRecord:
        return existing.model_copy(
            update={
                "content": candidate.content,
                "source": candidate.source,
                "version": existing.version + 1,
                "updated_at": datetime.now(
                    timezone.utc
                ),
                "metadata": candidate.metadata,
            }
        )

    @staticmethod
    def _merge_memories(
        candidate: CandidateMemory,
        existing: list[MemoryRecord],
    ) -> MemoryRecord:
        contents = [
            memory.content
            for memory in existing
        ]

        if candidate.content not in contents:
            contents.append(candidate.content)

        merged_content = " ".join(
            content.strip()
            for content in contents
            if content.strip()
        )

        latest_version = max(
            memory.version
            for memory in existing
        )

        source_history: list[dict[str, object]] = []

        for memory in existing:
            previous_history = memory.metadata.get("source_history")
            if isinstance(previous_history, list):
                source_history.extend(previous_history)
            else:
                source_history.append({
                    "thread_id": memory.source.thread_id,
                    "message_ids": list(memory.source.message_ids),
                })

        source_history.append({
            "thread_id": candidate.source.thread_id,
            "message_ids": list(candidate.source.message_ids),
        })

        metadata = dict(candidate.metadata)
        metadata["source_history"] = source_history

        return MemoryRecord(
            user_id=candidate.user_id,
            memory_type=candidate.memory_type,
            content=merged_content,
            status="active",
            source=candidate.source,
            version=latest_version + 1,
            metadata=metadata,
        )