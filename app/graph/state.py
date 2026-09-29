from typing import TypedDict

from app.contracts.context import Context
from app.contracts.retrieval import LTMResult, RAGResult, RetrievedItem, STMResult
from app.contracts.routing import RetrievalPlan
from app.contracts.tracing import ValidationResult


class GraphState(TypedDict):
    user_id: str
    thread_id: str
    query: str
    rewritten_query: str | None
    retrieval_plan: RetrievalPlan | None
    stm_result: STMResult | None
    ltm_result: LTMResult | None
    pdf_result: RAGResult | None
    merged_results: list[RetrievedItem]
    reranked_results: list[RetrievedItem]
    context: Context | None
    answer: str | None
    validation: ValidationResult | None
    retry_count: int
    trace_id: str
