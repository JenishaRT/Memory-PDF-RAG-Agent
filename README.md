# Production LangGraph Agent

Phase 0 skeleton for a modular conversational AI system with STM, LTM, PDF RAG, LangGraph orchestration, runtime/session management, persistence, tracing, and evaluation.

## Current status

Phase 0 only. Contracts, boundaries, configuration, JSONL conversation persistence, runtime skeleton, and CLI are implemented.

Actual STM retrieval, LTM retrieval/consolidation, PDF RAG retrieval, and the full LangGraph workflow are intentionally deferred.

## Run

```bash
python run.py --user user_001 --thread thread_001
```

Or:

```bash
python run.py --user user_001 --thread thread_001 --query "Hello"
```

## Test

```bash
pytest
```
