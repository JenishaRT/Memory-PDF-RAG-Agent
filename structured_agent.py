import pdfplumber
from pathlib import Path
from google import genai
import os
import sqlite3
import json
import time

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

pdf_folder = Path("pdfs")
db_path = "agent.db"


def setup_database():
    """Creates a new table just for structured results, if it doesn't exist yet"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS structured_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            tower_model TEXT,
            flow_rate_gpm TEXT,
            motor_horsepower TEXT,
            compliance_codes TEXT,
            warranty_years TEXT
        )
    """)
    conn.commit()
    conn.close()


def extract_text(pdf_file):
    """Same PDF text extraction you've always used"""
    all_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            all_text += (page.extract_text() or "") + "\n"
    return all_text


def extract_structured_fields(document_text, max_retries=3):
    """Asks Gemini to fill out a strict form, trying a fallback model if the primary is consistently busy"""
    prompt = f"""Read this document and extract ONLY the following fields.
Respond with ONLY valid JSON, no other text, no explanation, using exactly this shape:

{{
  "tower_model": "...",
  "flow_rate_gpm": "...",
  "motor_horsepower": "...",
  "compliance_codes": "...",
  "warranty_years": "..."
}}

If a field isn't mentioned in the document, use "not found" as its value.

Document:
{document_text}
"""

    models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash"]

    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                raw = response.text.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                return json.loads(raw)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    print(f"  {model_name} busy, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"  {model_name} failed after {max_retries} attempts, trying next model...")

    raise Exception("All models failed after retries")


def process_all_pdfs():
    setup_database()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for pdf_file in pdf_folder.glob("*.pdf"):
        cursor.execute("SELECT id FROM structured_documents WHERE filename = ?", (pdf_file.name,))
        if cursor.fetchone():
            print(f"Already processed: {pdf_file.name}")
            continue

        print(f"Processing {pdf_file.name}...")
        text = extract_text(pdf_file)

        try:
            fields = extract_structured_fields(text)
            cursor.execute("""
                INSERT INTO structured_documents
                (filename, tower_model, flow_rate_gpm, motor_horsepower, compliance_codes, warranty_years)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pdf_file.name,
                fields.get("tower_model"),
                fields.get("flow_rate_gpm"),
                fields.get("motor_horsepower"),
                fields.get("compliance_codes"),
                fields.get("warranty_years"),
            ))
            conn.commit()
            print(f"  Extracted: {fields}")
        except Exception as e:
            print(f"  Failed: {e}")

    conn.close()


if __name__ == "__main__":
    process_all_pdfs()
    print("\nDone! Check the structured_documents table in agent.db")