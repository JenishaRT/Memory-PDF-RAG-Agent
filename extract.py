import pdfplumber
from pathlib import Path
import json

pdf_folder = Path("pdfs")
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

for pdf_file in pdf_folder.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    all_text = ""
    all_tables = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                all_text += f"\n--- Page {page_number} ---\n{text}"

                tables = page.extract_tables()
                for table in tables:
                    all_tables.append({
                        "page": page_number,
                        "rows": table
                    })

        # Save the text
        text_path = output_folder / f"{pdf_file.stem}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(all_text)

        # Save the tables (only if any were found)
        if all_tables:
            tables_path = output_folder / f"{pdf_file.stem}_tables.json"
            with open(tables_path, "w", encoding="utf-8") as f:
                json.dump(all_tables, f, indent=2)
            print(f"  Saved text + {len(all_tables)} table(s)")
        else:
            print(f"  Saved text (no tables found)")

    except Exception as e:
        print(f"  Failed: {e}")

print("\nAll done!")