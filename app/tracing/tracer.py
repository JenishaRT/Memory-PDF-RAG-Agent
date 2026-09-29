from typing import Protocol

from pydantic import BaseModel


class TraceSink(Protocol):
    def emit(self, trace: BaseModel) -> None:
        ...


class NoOpTracer:
    """Phase 0 tracer that satisfies the boundary without persistence logic."""

    def emit(self, trace: BaseModel) -> None:
        return None
