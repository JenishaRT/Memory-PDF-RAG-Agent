from pathlib import Path

output_folder = Path("outputs")
query = input("Search for: ").lower()

found_anything = False

for txt_file in output_folder.glob("*.txt"):
    content = txt_file.read_text(encoding="utf-8")
    pages = content.split("--- Page ")

    for page_chunk in pages:
        lower_chunk = page_chunk.lower()
        if query in lower_chunk:
            page_label = page_chunk.split("\n")[0].strip(" -")
            found_anything = True

            # Find where the match is and grab ~150 characters around it
            position = lower_chunk.find(query)
            start = max(0, position - 75)
            end = min(len(page_chunk), position + 75)
            snippet = page_chunk[start:end].replace("\n", " ").strip()

            print(f"\n{txt_file.name}, page {page_label}")
            print(f"  ...{snippet}...")

if not found_anything:
    print("No matches found.")