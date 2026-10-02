import pdfplumber
from pathlib import Path
from google import genai
import os
import chromadb

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

pdf_folder = Path("pdfs")

# ChromaDB stores its data in a local folder, similar to how SQLite uses one file
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")


def chunk_text(text, chunk_size=1000):
    """Splits a long piece of text into smaller pieces of roughly chunk_size characters"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def get_embedding(text):
    """Converts a piece of text into its number-list representation"""
    result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)
    return result.embeddings[0].values


def process_pdfs():
    """Extracts, chunks, embeds, and stores every new PDF"""
    existing_ids = set(collection.get()["ids"])

    for pdf_file in pdf_folder.glob("*.pdf"):
        # Skip if we've already processed this file (checked by chunk ID prefix)
        if any(pdf_file.name in existing_id for existing_id in existing_ids):
            continue

        print(f"Processing {pdf_file.name}...")
        with pdfplumber.open(pdf_file) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += (page.extract_text() or "") + "\n"

        chunks = chunk_text(full_text)
        print(f"  Split into {len(chunks)} chunks, embedding each...")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            collection.add(
                ids=[f"{pdf_file.name}_chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": pdf_file.name}]
            )
        print(f"  Done.")


def ask_question(question):
    """Finds the most relevant chunks and asks Gemini using only those"""
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    relevant_chunks = results["documents"][0]
    sources = results["metadatas"][0]

    context = ""
    for chunk, meta in zip(relevant_chunks, sources):
        context += f"\n\n=== From {meta['source']} ===\n{chunk}"

    prompt = f"""Here are the most relevant document excerpts:

{context}

Based only on these excerpts, answer this question: {question}

If the answer isn't here, say so clearly."""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text


# --- Main program ---
print("Checking for new PDFs to process...\n")
process_pdfs()
print("\nReady! Ask questions using vector search.\n")

while True:
    question = input("Ask a question (or type 'quit' to exit): ")
    if question.lower() == "quit":
        break
    try:
        answer = ask_question(question)
        print("\n" + answer + "\n")
    except Exception as e:
        print(f"\nSomething went wrong: {e}\n")

print("Goodbye!")