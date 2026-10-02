"""
Automated PDF Ingestion + Structured Extraction
-------------------------------------------------
Point this at any PDF, and it will read through every page and break the
content into structured "chunks" automatically -- no manual typing needed.

For each chunk, it records:
  - chunk_id        : a unique ID, auto-generated
  - page_number     : which page it's on (1-indexed)
  - section_number  : if the chunk starts with a clause number like
                       "2.1.17." or "1.1.1", that number is captured here
                       (None if the chunk doesn't start with one)
  - text            : the actual text content of that chunk
  - bounding_region : the exact [x0, y0, x1, y1] box on the page, taken
                       directly from the PDF -- no unit guessing or
                       rotation correction needed, because it comes from
                       the same coordinate space PyMuPDF already uses for
                       highlighting.

The result is saved as a JSON file: "<pdf_name>_chunks.json". That file is
your automated "ingestion" output -- a structured map of the whole
document, ready to search against for any question, without retyping
anything by hand.

Usage:
    python ingest_pdf.py spec.pdf
    python ingest_pdf.py test.pdf

Install:
    pip install pymupdf --break-system-packages
"""

import fitz  # PyMuPDF
import re
import json
import sys
from pathlib import Path

# Matches clause numbers like "2.1.17.", "1.1", "3.1.2.1." at the start of
# a line of text.
SECTION_NUMBER_PATTERN = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.*)", re.DOTALL)

# Blocks shorter than this are usually noise (page numbers, stray marks)
# rather than real content, so they're skipped.
MIN_BLOCK_LENGTH = 8


def extract_chunks_from_page(page: "fitz.Page", page_number: int, start_id: int):
    """Pulls every meaningful text block out of one page, using PyMuPDF's
    own paragraph/block detection -- this naturally groups text the way
    it's laid out on the page, without needing to guess at formatting."""
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    chunks = []
    chunk_id = start_id

    for b in blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        text = text.strip()
        if len(text) < MIN_BLOCK_LENGTH:
            continue  # skip near-empty / noise blocks

        # collapse internal newlines/extra spaces into single spaces
        clean_text = re.sub(r"\s+", " ", text).strip()

        section_number = None
        match = SECTION_NUMBER_PATTERN.match(clean_text)
        if match:
            section_number = match.group(1).rstrip(".")

        chunks.append({
            "chunk_id": f"chunk-{chunk_id:04d}",
            "page_number": page_number,
            "section_number": section_number,
            "text": clean_text,
            "bounding_region": {
                "page_number": page_number,
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            },
        })
        chunk_id += 1

    return chunks, chunk_id


def ingest_pdf(file_path: str):
    """Reads through the whole PDF and returns the full list of chunks."""
    src = Path(file_path)
    doc = fitz.open(str(src))

    all_chunks = []
    next_id = 1
    for i, page in enumerate(doc):
        page_number = i + 1
        page_chunks, next_id = extract_chunks_from_page(page, page_number, next_id)
        all_chunks.extend(page_chunks)

    doc.close()

    output_path = src.with_name(f"{src.stem}_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    return all_chunks, output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <filename.pdf>")
        sys.exit(1)

    filename = sys.argv[1]
    chunks, output_path = ingest_pdf(filename)

    print(f"Ingested {filename}: found {len(chunks)} chunks across the document.")
    print(f"Saved to: {output_path}\n")

    print("First 3 chunks as a preview:")
    for c in chunks[:3]:
        preview = c["text"][:80] + ("..." if len(c["text"]) > 80 else "")
        print(f"  [{c['chunk_id']}] page {c['page_number']}, section {c['section_number']}: {preview}")