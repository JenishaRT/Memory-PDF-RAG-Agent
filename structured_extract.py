"""
Structured PDF Extraction
--------------------------
Goes beyond simple paragraph splitting: detects the document's actual
outline/hierarchy (PART 1, 1.1, 1.1.1, 2.1.17.1, etc.) and builds a
nested structure reflecting how the document is really organized --
not just a flat list of text blocks.

Produces two files:
  <name>_structured.json : the full nested outline (PART -> section ->
                            subsection -> ... ), showing the document's
                            real structure.
  <name>_chunks.json      : a flat list of content chunks, each tagged
                            with its full heading path (e.g.
                            "PART 2 GENERAL > 2.1 COOLING TOWER > 2.1.17"),
                            plus page number and exact bounding box --
                            ready to feed into highlight_all_sources.

Usage:
    python structured_extract.py spec.pdf

Install:
    pip install pymupdf --break-system-packages
"""

import fitz  # PyMuPDF
import re
import json
import sys
from pathlib import Path

PART_PATTERN = re.compile(r'^PART\s+(\d+)\s*[-\u2013]\s*(.+)$', re.IGNORECASE)
NUMBERED_PATTERN = re.compile(r'^(\d+(?:\.\d+){1,})\.?\s+(.*)$')
# A "spec code" like "23 65 16" (used in construction specs) looks similar
# to a heading number but is NOT part of the document's clause hierarchy --
# this pattern explicitly excludes it from being treated as a heading.
SPEC_CODE_PATTERN = re.compile(r'^\d{2}\s\d{2}\s\d{2}$')


def classify_line(text: str):
    """Looks at one line of text and decides whether it's a heading
    (PART or a numbered clause), and if so, at what nesting level.
    Returns None if the line is just ordinary content, not a heading."""
    text = text.strip()
    if SPEC_CODE_PATTERN.match(text):
        return None

    m = PART_PATTERN.match(text)
    if m:
        return {"level": 0, "number": f"PART {m.group(1)}", "title": m.group(2).strip()}

    m = NUMBERED_PATTERN.match(text)
    if m:
        number = m.group(1)
        depth = number.count(".")  # "1.1" -> level 1; "1.1.1" -> level 2; etc.
        return {"level": depth, "number": number, "title": m.group(2).strip()}

    return None


def extract_lines(page: "fitz.Page"):
    """Pulls every line of text on the page, in reading order, along with
    its exact bounding box -- using line-level detail (not just blocks),
    since headings and their surrounding text often share one block."""
    d = page.get_text("dict")
    lines = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans_text = "".join(s["text"] for s in line.get("spans", []))
            text = spans_text.strip()
            if not text:
                continue
            lines.append({"text": text, "bbox": list(line["bbox"])})
    return lines


def build_structure(doc: "fitz.Document"):
    """Walks every line on every page, in order, building the nested
    heading hierarchy plus a flat list of content chunks tagged with
    their full heading path."""
    root_sections = []
    stack = []  # stack[i] = the currently open heading node at level i
    chunk_id = 1
    flat_chunks = []

    def current_path():
        return " > ".join(f"{n['number']} {n['title']}".strip() for n in stack if n)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        for line in extract_lines(page):
            heading = classify_line(line["text"])

            if heading:
                level = heading["level"]
                node = {
                    "level": level,
                    "number": heading["number"],
                    "title": heading["title"],
                    "page_number": page_number,
                    "bbox": line["bbox"],
                    "children": [],
                }
                stack[:] = stack[:level]  # close any headings at this level or deeper
                parent = stack[level - 1] if (level > 0 and level - 1 < len(stack)) else None

                if level == 0 or parent is None:
                    root_sections.append(node)
                else:
                    parent["children"].append(node)

                while len(stack) <= level:
                    stack.append(None)
                stack[level] = node

            else:
                heading_path = current_path() or "(no heading yet)"
                flat_chunks.append({
                    "chunk_id": f"chunk-{chunk_id:04d}",
                    "page_number": page_number,
                    "heading_path": heading_path,
                    "text": line["text"],
                    "bounding_region": {"page_number": page_number, "bbox": line["bbox"]},
                })
                chunk_id += 1

    return root_sections, flat_chunks


def structured_extract(file_path: str):
    src = Path(file_path)
    doc = fitz.open(str(src))

    sections, chunks = build_structure(doc)
    doc.close()

    structure_path = src.with_name(f"{src.stem}_structured.json")
    chunks_path = src.with_name(f"{src.stem}_chunks.json")

    with open(structure_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return sections, chunks, structure_path, chunks_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python structured_extract.py <filename.pdf>")
        sys.exit(1)

    filename = sys.argv[1]
    sections, chunks, structure_path, chunks_path = structured_extract(filename)

    def count_headings(nodes):
        total = len(nodes)
        for n in nodes:
            total += count_headings(n["children"])
        return total

    print(f"Found {count_headings(sections)} headings (PART + numbered clauses).")
    print(f"Found {len(chunks)} content lines, each tagged with its heading path.")
    print(f"Structure saved to: {structure_path}")
    print(f"Chunks saved to:    {chunks_path}\n")

    print("Top-level sections found:")
    for s in sections:
        print(f"  {s['number']} - {s['title']}")

    print("\nFirst 3 content chunks as a preview (now tagged with real headings):")
    for c in chunks[:3]:
        preview = c["text"][:70] + ("..." if len(c["text"]) > 70 else "")
        print(f"  [{c['chunk_id']}] ({c['heading_path']}): {preview}")