"""
PDF Ingestion + Structured Extraction Pipeline (v3)
--------------------------------------------------
Adds four fixes based on real testing against a price-quote style PDF:

  1. Repeated-letterhead filter: text that appears identically on 3+
     different pages is recognized as boilerplate and never treated as
     a heading.
  2. Colon-or-ALL-CAPS requirement: a bold/large line only counts as a
     heading if it ends with ':' or is written in ALL CAPS.
  3. Real-letters filter: rejects anything without at least two letters
     in a row (bullet symbols, bare "(a)" markers).
  4. ALL-CAPS word-limit: an ALL-CAPS line with no colon and more than 7
     words is treated as a fragment of a caps-written legal paragraph,
     not a real heading (real headings are almost always short).

Usage:
    python pdf_pipeline.py --input spec.pdf
    python pdf_pipeline.py --input pdfs/
    python pdf_pipeline.py --input pdfs/ --output-dir results/

Install:
    pip install pymupdf --break-system-packages
"""

import fitz  # PyMuPDF
import re
import json
import sys
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

PART_PATTERN = re.compile(r'^PART\s+(\d+)\s*[-\u2013]\s*(.+)$', re.IGNORECASE)
SECTION_WORD_PATTERN = re.compile(r'^(SECTION|ARTICLE)\s+([\dIVXLC]+)\s*[-\u2013:]?\s*(.*)$', re.IGNORECASE)
NUMBERED_PATTERN = re.compile(r'^(\d+(?:\.\d+){1,})\.?\s+(.*)$')
SPEC_CODE_PATTERN = re.compile(r'^\d{2}\s\d{2}\s\d{2}$')
NUMERIC_OR_SYMBOL_ONLY = re.compile(r'^[\$\d,\.\s\-\u2013\u2014%()]+$')
HAS_LETTERS_PATTERN = re.compile(r'[A-Za-z]{2,}')

MAX_HEADING_WORDS = 10
MIN_TEXT_WORDS_FOR_STRUCTURE = 30
BOLD_FLAG_BIT = 1 << 4
HEADING_SIZE_RATIO = 1.15
REPEATED_PAGE_THRESHOLD = 3


def setup_logging(log_path: Path):
    logger = logging.getLogger("pdf_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def open_pdf_safe(path: Path, logger):
    warnings = []
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.error(f"  Could not open {path.name}: {e}")
        return None, [f"failed_to_open: {e}"]

    if doc.is_encrypted:
        try:
            ok = doc.authenticate("")
            if not ok:
                logger.error(f"  {path.name} is password-protected -- skipping.")
                doc.close()
                return None, ["encrypted_password_protected"]
            warnings.append("was_encrypted_but_opened_with_blank_password")
        except Exception as e:
            logger.error(f"  {path.name} is encrypted and could not be opened: {e}")
            doc.close()
            return None, [f"encrypted_open_failed: {e}"]

    if doc.page_count == 0:
        warnings.append("zero_pages")
        logger.warning(f"  {path.name} has zero pages.")

    return doc, warnings


def compute_body_font_size(doc: "fitz.Document", sample_pages: int = 10) -> float:
    sizes = Counter()
    pages_to_scan = min(sample_pages, len(doc))
    for i in range(pages_to_scan):
        d = doc[i].get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(span["size"])] += len(span["text"])
    if not sizes:
        return 10.0
    return sizes.most_common(1)[0][0]


def extract_lines(page: "fitz.Page"):
    d = page.get_text("dict")
    lines = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            max_size = max((s["size"] for s in spans), default=0)
            is_bold = any((s.get("flags", 0) & BOLD_FLAG_BIT) for s in spans)
            lines.append({
                "text": text,
                "bbox": list(line["bbox"]),
                "max_size": max_size,
                "is_bold": is_bold,
            })
    return lines


def find_repeated_lines(doc: "fitz.Document"):
    text_to_pages = defaultdict(set)
    for i in range(len(doc)):
        page_number = i + 1
        for line in extract_lines(doc[i]):
            text_to_pages[line["text"].strip()].add(page_number)

    return {text for text, pages in text_to_pages.items() if len(pages) >= REPEATED_PAGE_THRESHOLD}


def classify_line(line: dict, body_size: float, repeated_texts: set):
    text = line["text"].strip()

    if text in repeated_texts:
        return None

    if SPEC_CODE_PATTERN.match(text):
        return None

    m = PART_PATTERN.match(text)
    if m:
        return {"level": 0, "number": f"PART {m.group(1)}", "title": m.group(2).strip(), "source": "numbered"}

    m = SECTION_WORD_PATTERN.match(text)
    if m:
        return {"level": 0, "number": f"{m.group(1).upper()} {m.group(2)}", "title": m.group(3).strip(), "source": "numbered"}

    m = NUMBERED_PATTERN.match(text)
    if m:
        number = m.group(1)
        depth = number.count(".")
        return {"level": depth, "number": number, "title": m.group(2).strip(), "source": "numbered"}

    if not HAS_LETTERS_PATTERN.search(text):
        return None

    if NUMERIC_OR_SYMBOL_ONLY.match(text):
        return None

    word_count = len(text.split())

    if text.endswith(".") and word_count > 3:
        return None

    if word_count > MAX_HEADING_WORDS:
        return None

    looks_heading_shaped = text.endswith(":") or text.isupper()
    if not looks_heading_shaped:
        return None

    # An ALL-CAPS line with no colon and a lot of words is more likely a
    # fragment of a legal paragraph written in caps than a real heading --
    # real headings (even all-caps ones) are almost always short.
    if text.isupper() and not text.endswith(":") and word_count > 7:
        return None

    looks_like_heading = (
        len(text) <= 120
        and (line["max_size"] >= body_size * HEADING_SIZE_RATIO or
             (line["is_bold"] and line["max_size"] >= body_size))
    )
    if looks_like_heading:
        return {"level": 1, "number": "", "title": text, "source": "styling"}

    return None


def build_structure(doc: "fitz.Document", body_size: float, repeated_texts: set):
    root_sections = []
    stack = []
    chunk_id = 1
    flat_chunks = []
    total_words = 0

    def current_path():
        parts = []
        for n in stack:
            if not n:
                continue
            label = f"{n['number']} {n['title']}".strip() if n["number"] else n["title"]
            parts.append(label)
        return " > ".join(parts)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        lines = extract_lines(page)

        for line in lines:
            total_words += len(line["text"].split())
            heading = classify_line(line, body_size, repeated_texts)

            if heading:
                level = heading["level"]
                node = {
                    "level": level,
                    "number": heading["number"],
                    "title": heading["title"],
                    "detected_by": heading["source"],
                    "page_number": page_number,
                    "bbox": line["bbox"],
                    "children": [],
                }
                stack[:] = stack[:level]
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

    return root_sections, flat_chunks, total_words


def count_headings(nodes):
    total = len(nodes)
    for n in nodes:
        total += count_headings(n["children"])
    return total


def process_single_pdf(path: Path, output_dir: Path, logger):
    result = {
        "file": path.name,
        "status": "unknown",
        "warnings": [],
        "page_count": None,
        "headings_found": None,
        "chunks_found": None,
        "total_words": None,
        "structure_output": None,
        "chunks_output": None,
    }

    logger.info(f"Processing: {path.name}")
    doc, open_warnings = open_pdf_safe(path, logger)
    result["warnings"].extend(open_warnings)

    if doc is None:
        result["status"] = "failed_to_open"
        return result

    try:
        result["page_count"] = doc.page_count
        body_size = compute_body_font_size(doc)
        repeated_texts = find_repeated_lines(doc)
        logger.info(f"  Found {len(repeated_texts)} repeated letterhead/footer lines to ignore.")

        sections, chunks, total_words = build_structure(doc, body_size, repeated_texts)

        result["headings_found"] = count_headings(sections)
        result["chunks_found"] = len(chunks)
        result["total_words"] = total_words

        if total_words < MIN_TEXT_WORDS_FOR_STRUCTURE:
            result["warnings"].append(
                f"low_text_content: only {total_words} words found -- this PDF may be "
                f"a scanned image or CAD export with no real text layer."
            )
            logger.warning(f"  {path.name}: low text content ({total_words} words) -- flagged.")

        stem = path.stem
        structure_path = output_dir / f"{stem}_structured.json"
        chunks_path = output_dir / f"{stem}_chunks.json"

        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, indent=2)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        result["structure_output"] = str(structure_path)
        result["chunks_output"] = str(chunks_path)
        result["status"] = "success" if not any("low_text_content" in w for w in result["warnings"]) else "success_with_warnings"

        logger.info(
            f"  Done: {result['headings_found']} headings, {result['chunks_found']} chunks, "
            f"{total_words} words."
        )

    except Exception as e:
        logger.error(f"  Unexpected error processing {path.name}: {e}")
        result["status"] = "failed_during_processing"
        result["warnings"].append(str(e))
    finally:
        doc.close()

    return result


def get_pdf_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.pdf")) + sorted(input_path.glob("*.PDF"))
    return []


def main():
    parser = argparse.ArgumentParser(description="PDF ingestion + structured extraction pipeline")
    parser.add_argument("--input", required=True, help="A single PDF file, or a folder of PDFs")
    parser.add_argument("--output-dir", default="pipeline_output", help="Where to save results")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "pipeline_log.txt"
    logger = setup_logging(log_path)

    pdf_files = get_pdf_files(input_path)
    if not pdf_files:
        logger.error(f"No PDF files found at: {input_path}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process.")
    logger.info(f"Output directory: {output_dir}\n")

    results = []
    for pdf_path in pdf_files:
        result = process_single_pdf(pdf_path, output_dir, logger)
        results.append(result)

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "input_path": str(input_path),
            "total_files": len(results),
            "results": results,
        }, f, indent=2)

    succeeded = sum(1 for r in results if r["status"] in ("success", "success_with_warnings"))
    flagged = sum(1 for r in results if r["status"] == "success_with_warnings")
    failed = sum(1 for r in results if r["status"] not in ("success", "success_with_warnings"))

    logger.info("\n" + "=" * 60)
    logger.info(f"BATCH COMPLETE: {succeeded}/{len(results)} processed successfully")
    logger.info(f"  -> {flagged} flagged with warnings (e.g. low text content)")
    logger.info(f"  -> {failed} failed entirely")
    logger.info(f"Full summary: {summary_path}")
    logger.info(f"Full log:     {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()