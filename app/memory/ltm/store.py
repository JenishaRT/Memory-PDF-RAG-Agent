import json
from pathlib import Path
from typing import Protocol

from app.contracts.errors import MemoryError
from app.contracts.memory import MemoryRecord


class MemoryStore(Protocol):
    def get(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> MemoryRecord | None:
        ...

    def list(
        self,
        *,
        user_id: str,
        include_inactive: bool = False,
    ) -> list[MemoryRecord]:
        ...

    def save(
        self,
        memory: MemoryRecord,
    ) -> None:
        ...


class JsonMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def get(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> MemoryRecord | None:
        for memory in self._load():
            if (
                memory.user_id == user_id
                and memory.memory_id == memory_id
            ):
                return memory
        return None

    def list(
        self,
        *,
        user_id: str,
        include_inactive: bool = False,
    ) -> list[MemoryRecord]:
        memories = [
            memory
            for memory in self._load()
            if memory.user_id == user_id
        ]

        if not include_inactive:
            memories = [
                memory
                for memory in memories
                if memory.status == "active"
            ]

        return memories

    def save(
        self,
        memory: MemoryRecord,
    ) -> None:
        memories = self._load()

        updated = False

        for index, existing in enumerate(memories):
            if existing.memory_id != memory.memory_id:
                continue

            if existing.user_id != memory.user_id:
                raise MemoryError(
                    "Memory belongs to another user"
                )

            memories[index] = memory
            updated = True
            break

        if not updated:
            memories.append(memory)

        self._write(memories)

    def _load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryError(
                f"Failed to load memories: {self.path}"
            ) from exc

        if not isinstance(data, list):
            raise MemoryError(
                "Memory store must contain a JSON array"
            )

        try:
            return [
                MemoryRecord.model_validate(item)
                for item in data
            ]
        except Exception as exc:
            raise MemoryError(
                "Failed to validate stored memories"
            ) from exc

    def _write(
        self,
        memories: list[MemoryRecord],
    ) -> None:
        try:
            with self.path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    [
                        memory.model_dump(mode="json")
                        for memory in memories
                    ],
                    file,
                    indent=2,
                )
        except OSError as exc:
            raise MemoryError(
                f"Failed to save memories: {self.path}"
            ) from exc