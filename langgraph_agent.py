import pdfplumber
from pathlib import Path
from google import genai
import os
import sqlite3
import json
import time
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

pdf_folder = Path("pdfs")
db_path = "agent.db"

MODELS = ["gemini-3.6-flash", "gemini-2.5-flash"]
MAX_ATTEMPTS_PER_MODEL = 3

REQUIRED_FIELDS = [
    "tower_model",
    "flow_rate_gpm",
    "motor_horsepower",
    "compliance_codes",
    "warranty_years",
]


# ---------- STATE: the shared notebook every node reads and writes ----------
class ExtractionState(TypedDict):
    pdf_path: str
    filename: str
    text: str
    fields: Optional[dict]
    error: Optional[str]
    model_index: int
    attempt: int


# ---------- NODES: each one does exactly one job ----------
def extract_text_node(state: ExtractionState):
    """Reads the PDF and puts its text into the state"""
    print(f"Processing {state['filename']}...")
    all_text = ""
    with pdfplumber.open(state["pdf_path"]) as pdf:
        for page in pdf.pages:
            all_text += (page.extract_text() or "") + "\n"
    return {"text": all_text}


def llm_extract_node(state: ExtractionState):
    """Makes ONE attempt to get structured fields from Gemini"""
    model_name = MODELS[state["model_index"]]

    prompt = f"""Read this document and extract ONLY the following fields.
Respond with ONLY valid JSON, no other text, no explanation,
using exactly this shape:

{{
  "tower_model": "...",
  "flow_rate_gpm": "...",
  "motor_horsepower": "...",
  "compliance_codes": "...",
  "warranty_years": "..."
}}

If a field isn't mentioned in the document, use "not found" as its value.

Document:
{state['text']}
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return {"fields": json.loads(raw), "error": None}
    except Exception as e:
        return {"fields": None, "error": str(e)}


def to_number(value):
    """Pulls the first number out of text, so '9,200 gpm' becomes '9200'"""
    if value is None:
        return None
    match = re.search(r"\d[\d,]*\.?\d*", value)
    if match:
        return match.group(0).replace(",", "")
    return None


def validate_node(state: ExtractionState):
    """Checks the LLM output has every field, then cleans the values"""
    fields = state["fields"]

    if not isinstance(fields, dict):
        msg = "LLM response was not a JSON object"
        return {"fields": None, "error": msg}

    missing = [k for k in REQUIRED_FIELDS if k not in fields]
    if missing:
        msg = f"LLM response missing fields: {missing}"
        return {"fields": None, "error": msg}

    empty_words = ("not found", "n/a", "none", "null", "")
    cleaned = {}
    for key in REQUIRED_FIELDS:
        value = fields[key]
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        if value is not None:
            value = str(value).strip()
        if value is None or value.lower() in empty_words:
            value = None
        cleaned[key] = value

    # These columns should hold plain numbers, not text with units
    for key in ("flow_rate_gpm", "motor_horsepower", "warranty_years"):
        cleaned[key] = to_number(cleaned[key])

    return {"fields": cleaned, "error": None}


def retry_node(state: ExtractionState):
    """Waits, then moves to the next attempt (or the next model)"""
    attempt = state["attempt"] + 1
    model_index = state["model_index"]

    if attempt >= MAX_ATTEMPTS_PER_MODEL:
        model_index += 1
        attempt = 0
        print(f"  Switching to fallback model: {MODELS[model_index]}")

    wait_time = 15 * (attempt + 1)
    reason = (state["error"] or "unknown error")[:80]
    name = MODELS[model_index]
    print(f"  {name} problem ({reason}), waiting {wait_time}s...")
    time.sleep(wait_time)

    return {"attempt": attempt, "model_index": model_index}


def save_node(state: ExtractionState):
    """Saves the cleaned fields as one row in SQLite"""
    fields = state["fields"]
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO structured_documents
        (filename, tower_model, flow_rate_gpm, motor_horsepower,
         compliance_codes, warranty_years)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            state["filename"],
            fields["tower_model"],
            fields["flow_rate_gpm"],
            fields["motor_horsepower"],
            fields["compliance_codes"],
            fields["warranty_years"],
        ),
    )
    conn.commit()
    conn.close()
    print(f"  Saved: {fields}")
    return {}


def give_up_node(state: ExtractionState):
    print(f"  Failed permanently: {state['error']}")
    return {}


# ---------- ROUTERS: decide which arrow to follow ----------
def retry_or_give_up(state: ExtractionState):
    last_attempt = state["attempt"] + 1 >= MAX_ATTEMPTS_PER_MODEL
    last_model = state["model_index"] + 1 >= len(MODELS)

    if last_attempt and last_model:
        return "give_up"
    return "retry"


def route_after_llm(state: ExtractionState):
    if state["fields"] is not None:
        return "validate"
    return retry_or_give_up(state)


def route_after_validate(state: ExtractionState):
    if state["fields"] is not None:
        return "save"
    return retry_or_give_up(state)


# ---------- BUILD THE GRAPH: connect the boxes with arrows ----------
graph = StateGraph(ExtractionState)

graph.add_node("extract_text", extract_text_node)
graph.add_node("llm_extract", llm_extract_node)
graph.add_node("validate", validate_node)
graph.add_node("retry", retry_node)
graph.add_node("save", save_node)
graph.add_node("give_up", give_up_node)

graph.set_entry_point("extract_text")
graph.add_edge("extract_text", "llm_extract")
graph.add_conditional_edges(
    "llm_extract",
    route_after_llm,
    {"validate": "validate", "retry": "retry", "give_up": "give_up"},
)
graph.add_conditional_edges(
    "validate",
    route_after_validate,
    {"save": "save", "retry": "retry", "give_up": "give_up"},
)
graph.add_edge("retry", "llm_extract")
graph.add_edge("save", END)
graph.add_edge("give_up", END)

app = graph.compile()


# ---------- RUN IT: one trip through the graph per PDF ----------
def setup_database():
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS structured_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            tower_model TEXT,
            flow_rate_gpm TEXT,
            motor_horsepower TEXT,
            compliance_codes TEXT,
            warranty_years TEXT
        )
        """
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup_database()
    conn = sqlite3.connect(db_path)

    for pdf_file in pdf_folder.glob("*.pdf"):
        exists = conn.execute(
            "SELECT id FROM structured_documents WHERE filename = ?",
            (pdf_file.name,),
        ).fetchone()
        if exists:
            print(f"Already processed: {pdf_file.name}")
            continue

        app.invoke(
            {
                "pdf_path": str(pdf_file),
                "filename": pdf_file.name,
                "text": "",
                "fields": None,
                "error": None,
                "model_index": 0,
                "attempt": 0,
            }
        )

    conn.close()
    print("\nDone! Check the structured_documents table in agent.db")