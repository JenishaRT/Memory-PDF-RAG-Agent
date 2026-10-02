   # PDF Agent

A set of Python tools for pulling text, tables, document structure, and specific fields out of PDFs, then asking questions about them with Google Gemini.

The project grows in stages. It starts with plain text extraction and keyword search, then adds LLM Q&A, vector search over embeddings (RAG), structured field extraction into SQLite (including a LangGraph version with retries), and a heading-aware structure pipeline that doesn't need an LLM.

---

## Project layout

```
pdf_agent/
├── pdfs/                  # Put input PDFs here
├── outputs/               # Extracted text (.txt) and tables (_tables.json)
├── pipeline_output/       # Output of pdf_pipeline.py (structure, chunks, logs, summary)
├── chroma_db/             # ChromaDB vector store (created by vector_agent.py)
├── agent.db               # SQLite database (structured_documents table)
│
├── extract.py             # Extract text + tables from every PDF in pdfs/
├── search.py              # Keyword search over extracted text
├── ask.py                 # Ask one question over all extracted text (Gemini)
├── agent.py               # Interactive Q&A: extract new PDFs, then a question loop
├── vector_agent.py        # Interactive Q&A with embeddings + ChromaDB (RAG)
├── structured_agent.py    # LLM field extraction into SQLite
├── langgraph_agent.py     # Same field extraction as a LangGraph state machine
├── memory_agent.py        # LangGraph PDF Q&A with short- and long-term memory
├── stm.py                 # Short-term (session) memory helpers
├── ltm.py                 # Long-term memory store (long_term_memory table)
├── memory_utils.py        # Keyword helpers shared by stm.py and ltm.py
├── ingest_pdf.py          # Split a PDF into block-level chunks with bounding boxes
├── structured_extract.py  # Build a heading hierarchy + heading-tagged chunks
├── pdf_pipeline.py        # Batch structure pipeline (v3) with logging and heuristics
├── view_data.py           # Print rows from agent.db
├── delete_row.py          # Clear the structured_documents table
└── hello.py               # Sanity-check script
```

---

## Requirements

- Python 3.10+
- A Google Gemini API key (only for the LLM-based scripts)

Install the dependencies:

```bash
pip install pdfplumber pymupdf google-genai chromadb langgraph
```

| Package        | Used by                                                          |
|----------------|------------------------------------------------------------------|
| `pdfplumber`   | `extract.py`, `agent.py`, `vector_agent.py`, `structured_agent.py`, `langgraph_agent.py` |
| `pymupdf`      | `ingest_pdf.py`, `structured_extract.py`, `pdf_pipeline.py`      |
| `google-genai` | `ask.py`, `agent.py`, `vector_agent.py`, `structured_agent.py`, `langgraph_agent.py` |
| `chromadb`     | `vector_agent.py`                                                |
| `langgraph`    | `langgraph_agent.py`                                             |

### Set your API key

PowerShell:
```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

macOS/Linux:
```bash
export GEMINI_API_KEY="your-key-here"
```

---

## Usage

Run every script from the project root, because they use relative paths such as `pdfs/`, `outputs/`, and `agent.db`.

### 1. Text and table extraction

```bash
python extract.py
```
For every PDF in `pdfs/`, this writes:
- `outputs/<name>.txt`: the full text, with `--- Page N ---` markers
- `outputs/<name>_tables.json`: the tables it detected, with page numbers (only written when tables exist)

### 2. Keyword search (no LLM)

```bash
python search.py
```
Prompts for a search term and prints each matching file and page with about 150 characters of surrounding context.

### 3. Ask questions with Gemini

**One question:**
```bash
python ask.py
```

**Interactive loop:** extracts any new PDFs first, then lets you ask questions until you type `quit`:
```bash
python agent.py
```

Both scripts send **all** extracted text to Gemini (`gemini-3.6-flash`), so they work best on a small number of documents.

### 4. Vector search Q&A (RAG)

```bash
python vector_agent.py
```
- Splits each new PDF into chunks of about 1,000 characters
- Embeds each chunk with `gemini-embedding-001` and stores it in `chroma_db/`
- For each question, retrieves the 5 most relevant chunks and answers from those alone

This scales to larger document sets better than `agent.py`. It skips PDFs that were already embedded.

### 5. Structured field extraction into SQLite

Both scripts extract these fields from every PDF in `pdfs/` into the `structured_documents` table in `agent.db`:

| Field              | Description                         |
|--------------------|-------------------------------------|
| `tower_model`      | Cooling tower model                 |
| `flow_rate_gpm`    | Flow rate in GPM                    |
| `motor_horsepower` | Motor horsepower                    |
| `compliance_codes` | Compliance codes / standards        |
| `warranty_years`   | Warranty period in years            |

**Simple version:**
```bash
python structured_agent.py
```

**LangGraph version (recommended):**
```bash
python langgraph_agent.py
```

The LangGraph agent runs as a state machine:

```
extract_text → llm_extract → validate → save → END
                    ↑            │
                    └── retry ←──┤
                                 └→ give_up → END
```

- **validate** checks that every field is present and normalizes values. It turns `"not found"`/`"n/a"` into `NULL`, joins lists into strings, and reduces numeric fields to plain numbers (`"9,200 gpm"` becomes `9200`).
- **retry** waits with increasing backoff (15s, 30s, …), makes up to 3 attempts per model, then falls back from `gemini-3.6-flash` to `gemini-2.5-flash`.
- PDFs already in the database are skipped.

**View or reset the results:**
```bash
python view_data.py     # print all rows
python delete_row.py    # deletes ALL rows so every PDF is reprocessed
```

### 6. Q&A with short-term and long-term memory

```bash
python memory_agent.py "pdfs/SPX_NYU_770 Broadway_BOD CT_8-20-26 MM.pdf"   # interactive
python memory_agent.py --demo                                              # scripted demo
python memory_agent.py --show-ltm                                          # print LTM
```

In interactive mode, `stm` prints this session's memory, `ltm` prints long-term memory, `new` starts a fresh session (empty STM, same LTM), and `quit` exits.

This agent answers questions about a PDF and remembers context. It reuses the text extraction, field extraction, validation, and retry nodes from `langgraph_agent.py` without changing them.

```
load_pdf → doc_fields → stm_check ──(STM enough)──→ build_context → llm_answer → validate_answer → answer → update_stm → update_ltm → END
                            └──(not enough)→ ltm_check ──┘              ↑               │
                                                                        └──── retry ←───┴→ give_up → END
```

| Memory | File | Where it lives | Lifetime |
|---|---|---|---|
| **STM** | `stm.py` | `stm_turns` in the graph state, kept per session by a LangGraph `MemorySaver` checkpointer | Until the program exits or you type `new` |
| **LTM** | `ltm.py` | `long_term_memory` table in `agent.db` | Permanent |

- **STM** stores the last 10 questions and answers, so follow-ups like "what about the pump?" work.
- **LTM** is checked only when STM has nothing relevant, or when the question mentions an earlier session ("last time", "previously").
- The prompt ranks the sources **current PDF > STM > LTM**. Every answer is labeled with the source it came from (`pdf`, `stm`, `ltm`, or `none`). Validation rejects an answer that cites a source the model wasn't given.
- **What goes into LTM:**
  - Things the user explicitly asks to remember ("remember…", "I prefer…")
  - The structured fields of each newly opened PDF
  - PDF answers whose quoted evidence was found word for word in the PDF text
- **What stays out of LTM:** answers taken from memory, "not found" answers, and answers with unverified evidence.
- Each fact is unique by (category, PDF, key). Saving the same fact again does nothing, and saving a new value for the same key replaces the old one.

### 7. Document structure extraction (no LLM)

These scripts use PyMuPDF to recover the document's layout and outline. Every chunk keeps its page number and exact bounding box, so it can be traced back to the source or highlighted.

**Block-level chunks:**
```bash
python ingest_pdf.py pdfs/Spec.pdf
```
Writes `<name>_chunks.json` next to the PDF. Each chunk has `chunk_id`, `page_number`, `section_number` (e.g. `2.1.17`), `text`, and `bounding_region`.

**Heading hierarchy (numbered specs):**
```bash
python structured_extract.py pdfs/Spec.pdf
```
Detects `PART N - TITLE` and numbered clauses (`1.1`, `2.1.17.1`, …) and writes:
- `<name>_structured.json`: the nested outline (PART → section → subsection …)
- `<name>_chunks.json`: line-level chunks, each tagged with a `heading_path` such as `PART 2 PRODUCTS > 2.1 COOLING TOWER > 2.1.17 ...`

**Batch pipeline (v3, most robust):**
```bash
python pdf_pipeline.py --input pdfs/Spec.pdf
python pdf_pipeline.py --input pdfs/
python pdf_pipeline.py --input pdfs/ --output-dir results/
```

It does everything `structured_extract.py` does, plus:
- Detects `SECTION`/`ARTICLE` headings and headings identified by styling (bold or larger font)
- Ignores repeated letterheads and footers that appear on 3 or more pages
- Requires styled headings to end with `:` or be written in ALL CAPS
- Rejects symbol-only lines, bare markers, and long ALL-CAPS legal text
- Handles encrypted and empty PDFs, and flags low-text PDFs (likely scanned or CAD exports)

Output in `pipeline_output/` (or `--output-dir`):
- `<name>_structured.json` and `<name>_chunks.json`
- `batch_summary.json`: per-file status, page/heading/chunk/word counts, and warnings
- `pipeline_log.txt`: the full run log

---

## Choosing a tool

| Goal                                         | Use                          |
|----------------------------------------------|------------------------------|
| Get raw text/tables out of PDFs              | `extract.py`                 |
| Find where a word appears                    | `search.py`                  |
| Ask questions about a few PDFs               | `agent.py`                   |
| Ask questions across many or large PDFs      | `vector_agent.py`            |
| Pull specific fields into a database         | `langgraph_agent.py`         |
| Get the document outline with source locations | `pdf_pipeline.py`          |

---

## Notes and limitations

- Scanned (image-only) PDFs have no text layer. These scripts don't run OCR, so they extract little or nothing. `pdf_pipeline.py` flags them as `low_text_content`.
- The field schema in `structured_agent.py` and `langgraph_agent.py` is specific to cooling towers. To extract other fields, edit `REQUIRED_FIELDS`, the prompt, and the table definition.
- `agent.py` and `ask.py` send whole documents to the model, which can hit context or rate limits with many PDFs. Use `vector_agent.py` instead in that case.
- `vector_agent.py` uses fixed-size character chunks. A smarter option is to embed the heading-tagged chunks from `pdf_pipeline.py`.
- Despite its name, `delete_row.py` clears the whole `structured_documents` table.
