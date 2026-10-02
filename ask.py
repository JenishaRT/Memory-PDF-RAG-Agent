from pathlib import Path
from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

output_folder = Path("outputs")

all_documents = ""
for txt_file in output_folder.glob("*.txt"):
    all_documents += f"\n\n=== Document: {txt_file.name} ===\n{txt_file.read_text(encoding='utf-8')}"

question = input("Ask a question about your documents: ")

prompt = f"""Here are extracted documents:

{all_documents}

Based only on these documents, answer this question: {question}

If the answer isn't in the documents, say so clearly."""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\n" + response.text)