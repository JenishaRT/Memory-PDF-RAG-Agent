"""Question-answering agent over a PDF, with short-term and long-term memory.

Flow for every question:

    load_pdf -> doc_fields -> stm_check --(STM enough)--> build_context
                                  |                            ^
                                  +--(not enough)--> ltm_check-+
    build_context -> llm_answer -> validate_answer -> answer -> update_stm -> update_ltm -> END
                         ^              |
                         +--- retry <---+--> give_up -> END

The PDF text extraction, the field extraction, the field validation and the
retry/fallback logic are imported unchanged from langgraph_agent.py.
"""
import argparse
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

import ltm
import stm
from langgraph_agent import (
    MODELS,
    REQUIRED_FIELDS,
    client,
    db_path,
    extract_text_node,
    llm_extract_node,
    retry_node,
    retry_or_give_up,
    to_number,
    validate_node,
)
from memory_utils import normalize_space

long_term_memory = ltm.LongTermMemory(db_path)

ANSWER_FIELDS = [
    "resolved_query", "answer", "entity", "attribute", "value", "source", "evidence",
]
VALID_SOURCES = ("pdf", "stm", "ltm", "none")
EMPTY_VALUES = ("not found", "n/a", "none", "null", "unknown", "")


# ---------- STATE ----------
class MemoryAgentState(TypedDict, total=False):
    # Input for this turn
    query: str
    pdf_path: Optional[str]
    filename: Optional[str]

    # Current PDF (kept between turns by the checkpointer)
    text: str
    loaded_pdf: Optional[str]
    doc_fields: Optional[dict]    # output of the existing structured extraction
    doc_fields_for: Optional[str]
    doc_fields_new: bool          # True the first time a PDF's fields are read

    # Short-term memory
    stm_turns: list               # every turn of this session (kept between turns)
    stm_context: list             # the turns relevant to this query
    stm_sufficient: bool

    # Long-term memory
    ltm_context: list             # LTM rows relevant to this query
    ltm_error: Optional[str]
    ltm_updates: list             # what update_ltm did, for logging

    # LLM call, validation and retries (same keys as langgraph_agent.py)
    prompt: str
    result: Optional[dict]
    error: Optional[str]
    model_index: int
    attempt: int

    # Output
    answer: Optional[str]
    sources: Optional[dict]


# ---------- NODES: current PDF ----------
def load_pdf_node(state: MemoryAgentState):
    """Extracts the PDF text, but only when the PDF changed since the last turn"""
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        print("  [PDF] No PDF for this question, answering from memory only")
        return {"text": "", "loaded_pdf": None}

    if pdf_path == state.get("loaded_pdf") and state.get("text"):
        return {}

    if not Path(pdf_path).exists():
        print(f"  [PDF] File not found: {pdf_path}, answering from memory only")
        return {"text": "", "loaded_pdf": None, "filename": None}

    result = extract_text_node(state)
    return {"text": result["text"], "loaded_pdf": pdf_path}


def load_saved_fields(filename):
    """Reads the fields langgraph_agent.py already saved for this PDF, if any"""
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                f"SELECT {', '.join(REQUIRED_FIELDS)} FROM structured_documents "
                "WHERE filename = ?",
                (filename,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return dict(zip(REQUIRED_FIELDS, row)) if row else None


def run_existing_extraction(text):
    """Runs the unchanged extraction + validation nodes, one try per model"""
    for model_index in range(len(MODELS)):
        extracted = llm_extract_node({"text": text, "model_index": model_index})
        if extracted["fields"] is None:
            continue
        validated = validate_node({"fields": extracted["fields"]})
        if validated["fields"] is not None:
            return validated["fields"]
    return None


def doc_fields_node(state: MemoryAgentState):
    """Gets the structured fields for the current PDF, once per PDF"""
    filename = state.get("filename")
    if not filename or not state.get("text"):
        return {"doc_fields": None, "doc_fields_for": None, "doc_fields_new": False}
    if state.get("doc_fields_for") == filename:
        return {"doc_fields_new": False}

    fields = load_saved_fields(filename)
    if fields is None:
        print("  [PDF] Running structured extraction for this PDF...")
        fields = run_existing_extraction(state["text"])

    return {
        "doc_fields": fields,
        "doc_fields_for": filename,
        "doc_fields_new": fields is not None,
    }


# ---------- NODES: memory checks ----------
def stm_check_node(state: MemoryAgentState):
    """Looks for relevant turns earlier in this session"""
    query = state["query"]
    relevant = stm.find_relevant_turns(state.get("stm_turns", []), query)

    # STM is enough unless it's empty or the user asks about an earlier session
    sufficient = bool(relevant) and not ltm.asks_about_past_sessions(query)

    if relevant:
        numbers = ", ".join(f"STM-{t['turn']}" for t in relevant)
        print(f"  [STM] Found {len(relevant)} relevant turn(s): {numbers}")
    else:
        print("  [STM] Nothing relevant in this session")
    return {"stm_context": relevant, "stm_sufficient": sufficient}


def ltm_check_node(state: MemoryAgentState):
    """Looks for relevant facts saved in earlier sessions"""
    try:
        records = long_term_memory.search(state["query"], state.get("filename"))
    except Exception as e:
        # A broken memory store must never stop the agent from answering
        print(f"  [LTM] Retrieval failed ({e}), continuing without LTM")
        return {"ltm_context": [], "ltm_error": str(e)}

    if records:
        numbers = ", ".join(f"LTM-{r['id']}" for r in records)
        print(f"  [LTM] Found {len(records)} relevant memory(s): {numbers}")
    else:
        print("  [LTM] Nothing relevant from earlier sessions")
    return {"ltm_context": records, "ltm_error": None}


def build_context_node(state: MemoryAgentState):
    """Combines the current PDF, STM and LTM into one prompt, PDF first"""
    filename = state.get("filename")
    text = state.get("text")

    if text:
        fields = state.get("doc_fields")
        fields_text = json.dumps(fields, indent=2) if fields else "(not available)"
        pdf_section = (
            f"Key fields already extracted from this PDF:\n{fields_text}\n\n"
            f"Full text:\n{text}"
        )
    else:
        filename = "none"
        pdf_section = "(no PDF is loaded for this question)"

    prompt = f"""You answer questions about engineering PDFs.

Use the information below in this priority order:
1. CURRENT PDF: the source of truth for anything about this document.
2. SHORT-TERM MEMORY: earlier turns of this conversation. Use it to understand
   follow-ups such as "what about the pump?", "that value", "the previous answer".
   A follow-up usually asks for the same property as the previous question,
   about a different item.
3. LONG-TERM MEMORY: facts saved in earlier sessions. Use it only when the
   current PDF does not contain the answer. If it disagrees with the current
   PDF, trust the PDF.

Respond with ONLY valid JSON, no other text, no explanation,
using exactly this shape:

{{
  "resolved_query": "the question rewritten as a complete stand-alone question",
  "answer": "a short, direct answer in one or two sentences",
  "entity": "the item asked about, short and lowercase, e.g. cooling tower fan",
  "attribute": "the property asked about, short and lowercase, e.g. motor horsepower",
  "value": "the value with its unit, e.g. 30 HP, or not found",
  "source": "pdf, stm, ltm or none",
  "evidence": "an exact quote (under 200 characters) from that source, or empty"
}}

Rules for "source", which says where the answer itself came from:
- "pdf": found in the CURRENT PDF. Copy "evidence" word for word from the PDF text.
- "stm": only in SHORT-TERM MEMORY.
- "ltm": only in LONG-TERM MEMORY. Put the memory text in "evidence".
- "none": nothing above contains it. Say so in "answer" and use "not found" as value.
If the user only asks you to remember something, acknowledge it and use "none".

=== CURRENT PDF: {filename} ===
{pdf_section}

=== SHORT-TERM MEMORY (this conversation) ===
{stm.format_turns(state.get("stm_context", []))}

=== LONG-TERM MEMORY (earlier sessions) ===
{ltm.format_records(state.get("ltm_context", []))}

=== USER QUESTION ===
{state["query"]}
"""
    return {"prompt": prompt}


# ---------- NODES: LLM, validation, answer ----------
def llm_answer_node(state: MemoryAgentState):
    """Makes ONE attempt to get a structured JSON answer from Gemini"""
    model_name = MODELS[state["model_index"]]
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=state["prompt"],
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return {"result": json.loads(raw), "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def validate_answer_node(state: MemoryAgentState):
    """Checks the JSON answer and that its claimed source really was available"""
    result = state["result"]
    if not isinstance(result, dict):
        return {"result": None, "error": "LLM response was not a JSON object"}

    missing = [k for k in ANSWER_FIELDS if k not in result]
    if missing:
        return {"result": None, "error": f"LLM response missing fields: {missing}"}

    cleaned = {k: str(result[k] if result[k] is not None else "").strip() for k in ANSWER_FIELDS}
    source = cleaned["source"].lower()
    if source not in VALID_SOURCES:
        return {"result": None, "error": f"Unknown source: {source!r}"}

    # The model may only cite something we actually gave it
    available = {
        "pdf": bool(state.get("text")),
        "stm": bool(state.get("stm_context")),
        "ltm": bool(state.get("ltm_context")),
        "none": True,
    }
    if not available[source]:
        return {"result": None, "error": f"Answer cites {source} but none was provided"}

    evidence = normalize_space(cleaned["evidence"])
    cleaned["source"] = source
    cleaned["found"] = source != "none" and cleaned["value"].lower() not in EMPTY_VALUES
    cleaned["value_number"] = to_number(cleaned["value"])
    cleaned["evidence_verified"] = (
        source == "pdf"
        and bool(evidence)
        and evidence in normalize_space(state["text"])
    )
    return {"result": cleaned, "error": None}


SOURCE_LABELS = {
    "pdf": "Current PDF",
    "stm": "Short-term memory (this session)",
    "ltm": "Long-term memory (earlier session)",
    "none": "Not found in the PDF or memory",
}


def answer_node(state: MemoryAgentState):
    """Builds the final answer and records which source each piece came from"""
    result = state["result"]
    source = result["source"]
    label = SOURCE_LABELS[source]
    if source == "pdf":
        label += f" ({state['filename']})"

    sources = {
        "answer_source": source,
        "answer_source_label": label,
        "evidence": result["evidence"] or None,
        "evidence_verified_in_pdf": result["evidence_verified"],
        "context_given": {
            "pdf": state.get("filename") if state.get("text") else None,
            "stm": [f"STM-{t['turn']}" for t in state.get("stm_context", [])],
            "ltm": [f"LTM-{r['id']}" for r in state.get("ltm_context", [])],
        },
    }

    print(f"\nAnswer: {result['answer']}")
    print(f"Source: {label}")
    if result["evidence"]:
        check = " (verified in PDF)" if result["evidence_verified"] else ""
        print(f"Evidence: \"{result['evidence']}\"{check}")
    return {"answer": result["answer"], "sources": sources}


def give_up_node(state: MemoryAgentState):
    message = "Sorry, I couldn't get a valid answer from the model. Please try again."
    print(f"  Failed permanently: {state['error']}")
    print(f"\nAnswer: {message}")
    return {"answer": message, "sources": None}


# ---------- NODES: memory updates ----------
def update_stm_node(state: MemoryAgentState):
    """Adds this question and answer to the session's STM"""
    result = state["result"]
    turn = {
        "query": state["query"],
        "resolved_query": result["resolved_query"],
        "answer": result["answer"],
        "entity": result["entity"],
        "attribute": result["attribute"],
        "value": result["value"],
        "source": result["source"],
        "filename": state.get("filename"),
    }
    turns = stm.add_turn(state.get("stm_turns", []), turn)
    print(f"  [STM] Saved as STM-{turns[-1]['turn']} ({len(turns)} turn(s) in this session)")
    return {"stm_turns": turns}


def save_to_ltm(updates, category, scope, key, value, source):
    status, old_value = long_term_memory.save(category, scope, key, value, source)
    if status == "updated":
        updates.append(f"updated '{key}': {old_value} -> {value}")
    elif status == "inserted":
        updates.append(f"saved '{key}' = {value}")


def update_ltm_node(state: MemoryAgentState):
    """Saves only what is worth keeping across sessions.

    Saved:
      1. Anything the user explicitly asks to remember (as a preference).
      2. The structured fields of a newly opened PDF.
      3. An answer found in the current PDF whose evidence quote was verified.
    Not saved: answers taken from STM or LTM (nothing new), "not found"
    answers, and PDF answers whose quote couldn't be found in the PDF text.
    """
    result = state["result"]
    filename = state.get("filename")
    updates = []

    try:
        if ltm.wants_to_remember(state["query"]):
            save_to_ltm(updates, "preference", "user", state["query"], state["query"], "user")

        if state.get("doc_fields_new"):
            for field, value in state["doc_fields"].items():
                if value:
                    save_to_ltm(
                        updates, "document_fact", filename,
                        field.replace("_", " "), value, "structured_extraction",
                    )

        if result["source"] == "pdf" and result["found"] and result["evidence_verified"]:
            save_to_ltm(
                updates, "document_fact", filename,
                f"{result['entity']} | {result['attribute']}", result["value"], "pdf_answer",
            )
    except Exception as e:
        print(f"  [LTM] Update failed ({e}), the answer is unaffected")
        return {"ltm_updates": [f"error: {e}"]}

    if updates:
        for update in updates:
            print(f"  [LTM] {update}")
    else:
        print("  [LTM] Nothing new worth keeping")
    return {"ltm_updates": updates}


# ---------- ROUTERS ----------
def route_after_stm(state: MemoryAgentState):
    if state["stm_sufficient"]:
        return "build_context"
    return "ltm_check"


def route_after_llm(state: MemoryAgentState):
    if state["result"] is not None:
        return "validate_answer"
    return retry_or_give_up(state)


def route_after_validate(state: MemoryAgentState):
    if state["result"] is not None:
        return "answer"
    return retry_or_give_up(state)


# ---------- BUILD THE GRAPH ----------
graph = StateGraph(MemoryAgentState)

graph.add_node("load_pdf", load_pdf_node)
graph.add_node("doc_fields", doc_fields_node)
graph.add_node("stm_check", stm_check_node)
graph.add_node("ltm_check", ltm_check_node)
graph.add_node("build_context", build_context_node)
graph.add_node("llm_answer", llm_answer_node)
graph.add_node("validate_answer", validate_answer_node)
graph.add_node("retry", retry_node)
graph.add_node("answer", answer_node)
graph.add_node("give_up", give_up_node)
graph.add_node("update_stm", update_stm_node)
graph.add_node("update_ltm", update_ltm_node)

graph.set_entry_point("load_pdf")
graph.add_edge("load_pdf", "doc_fields")
graph.add_edge("doc_fields", "stm_check")
graph.add_conditional_edges(
    "stm_check",
    route_after_stm,
    {"build_context": "build_context", "ltm_check": "ltm_check"},
)
graph.add_edge("ltm_check", "build_context")
graph.add_edge("build_context", "llm_answer")
graph.add_conditional_edges(
    "llm_answer",
    route_after_llm,
    {"validate_answer": "validate_answer", "retry": "retry", "give_up": "give_up"},
)
graph.add_conditional_edges(
    "validate_answer",
    route_after_validate,
    {"answer": "answer", "retry": "retry", "give_up": "give_up"},
)
graph.add_edge("retry", "llm_answer")
graph.add_edge("answer", "update_stm")
graph.add_edge("update_stm", "update_ltm")
graph.add_edge("update_ltm", END)
graph.add_edge("give_up", END)

# The checkpointer keeps each session's state (including stm_turns) in memory,
# keyed by thread_id. A new thread_id means a new session with empty STM.
app = graph.compile(checkpointer=MemorySaver())


# ---------- RUN IT ----------
def ask(session_id, query, pdf_path=None):
    """Runs one question through the graph for the given session"""
    print(f"\n> {query}")
    turn_input = {
        "query": query,
        "pdf_path": pdf_path,
        "filename": Path(pdf_path).name if pdf_path else None,
        # Reset the per-turn values; text, doc_fields and stm_turns carry over
        "stm_context": [],
        "stm_sufficient": False,
        "ltm_context": [],
        "ltm_error": None,
        "ltm_updates": [],
        "result": None,
        "error": None,
        "model_index": 0,
        "attempt": 0,
        "answer": None,
        "sources": None,
    }
    config = {"configurable": {"thread_id": session_id}}
    return app.invoke(turn_input, config)


def show_stm(session_id):
    config = {"configurable": {"thread_id": session_id}}
    turns = app.get_state(config).values.get("stm_turns", [])
    print(stm.format_turns(turns) if turns else "STM is empty")


def show_ltm():
    records = long_term_memory.all()
    if not records:
        print("LTM is empty")
    for r in records:
        print(f"[LTM-{r['id']}] {r['category']:<13} {r['scope']}: {r['mem_key']} = {r['value']}")


def run_demo(pdf_path):
    """Session 1 uses the PDF and STM; session 2 starts fresh and uses LTM"""
    print("=" * 20, "SESSION 1 (PDF loaded)", "=" * 20)
    session_1 = f"demo-{uuid.uuid4().hex[:8]}"
    ask(session_1, "What is the HP of the cooling tower fan?", pdf_path)
    ask(session_1, "What about the basin heater?", pdf_path)
    ask(session_1, "What about the pump?", pdf_path)
    ask(session_1, "Remember that I prefer answers in both HP and kW.", pdf_path)

    print("\n" + "=" * 20, "SESSION 2 (new session, no PDF)", "=" * 20)
    session_2 = f"demo-{uuid.uuid4().hex[:8]}"
    ask(session_2, "What was the fan motor HP for the 770 Broadway cooling tower?")


def interactive(pdf_path, session_id):
    print(f"Session: {session_id}")
    print("Commands: 'stm' shows this session, 'ltm' shows long-term memory,")
    print("          'new' starts a new session (empty STM), 'quit' exits.")
    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            continue
        command = query.lower()
        if command in ("quit", "exit"):
            break
        if command == "stm":
            show_stm(session_id)
        elif command == "ltm":
            show_ltm()
        elif command == "new":
            session_id = uuid.uuid4().hex[:8]
            print(f"New session: {session_id} (STM cleared, LTM kept)")
        else:
            ask(session_id, query, pdf_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Q&A agent with STM and LTM")
    parser.add_argument("pdf", nargs="?", help="PDF to ask about (optional)")
    parser.add_argument("--session", default=None, help="session id (default: random)")
    parser.add_argument("--demo", action="store_true", help="run the scripted demo")
    parser.add_argument("--show-ltm", action="store_true", help="print LTM and exit")
    args = parser.parse_args()

    if args.show_ltm:
        show_ltm()
    elif args.demo:
        run_demo(args.pdf or "pdfs/SPX_NYU_770 Broadway_BOD CT_8-20-26 MM.pdf")
    else:
        interactive(args.pdf, args.session or uuid.uuid4().hex[:8])
