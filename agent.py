import pdfplumber
from pathlib import Path
from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

pdf_folder = Path("pdfs")
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)


def extract_all_pdfs():
    """Reads every PDF in the pdfs folder and saves its text to outputs/"""
    for pdf_file in pdf_folder.glob("*.pdf"):
        text_path = output_folder / f"{pdf_file.stem}.txt"

        # Skip files we've already extracted, so re-running is fast
        if text_path.exists():
            continue

        print(f"Extracting {pdf_file.name}...")
        all_text = ""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    all_text += f"\n--- Page {page_number} ---\n{text}"
            text_path.write_text(all_text, encoding="utf-8")
        except Exception as e:
            print(f"  Failed: {e}")


def ask_question(question):
    """Sends all extracted text + a question to Gemini and returns the answer"""
    all_documents = ""
    for txt_file in output_folder.glob("*.txt"):
        all_documents += f"\n\n=== Document: {txt_file.name} ===\n{txt_file.read_text(encoding='utf-8')}"

    prompt = f"""Here are extracted documents:

{all_documents}

Based only on these documents, answer this question: {question}

If the answer isn't in the documents, say so clearly."""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text


# --- Main program ---
print("Checking for new PDFs to process...\n")
extract_all_pdfs()
print("\nReady! All documents are extracted and indexed.\n")

import time

while True:
    question = input("Ask a question (or type 'quit' to exit): ")
    if question.lower() == "quit":
        break

    try:
        answer = ask_question(question)
        print("\n" + answer + "\n")
    except Exception as e:
        print(f"\nSomething went wrong (likely temporary server load). Try again in a moment.")
        print(f"Details: {e}\n")

print("Goodbye!")