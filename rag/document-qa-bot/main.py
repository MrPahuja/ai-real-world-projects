"""
Document Q&A Bot with RAG
=========================
Ingests PDFs, indexes them in ChromaDB, and answers questions
with source citations and streaming responses.

Setup:
    pip install -r requirements.txt
    cp .env.example .env  # add OPENAI_API_KEY
    python main.py

Usage:
    The Gradio UI launches at http://localhost:7860
"""

import os
from pathlib import Path
from typing import Generator

import chromadb
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


# ── Indexing ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(pdf_path: str) -> int:
    """Ingest a PDF file into ChromaDB. Returns the number of chunks indexed."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full_text)

    embeddings = client.embeddings.create(input=chunks, model=EMBED_MODEL).data
    filename = Path(pdf_path).name

    collection.add(
        documents=chunks,
        embeddings=[e.embedding for e in embeddings],
        ids=[f"{filename}::chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


# ── Retrieval + Generation ───────────────────────────────────────────────────

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Retrieve the top-K most relevant chunks for a question."""
    q_embedding = client.embeddings.create(input=question, model=EMBED_MODEL).data[0].embedding
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
    return [
        {"text": doc, "source": meta["source"], "chunk": meta["chunk"], "score": 1 - dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def answer_stream(question: str, top_k: int = 5) -> Generator[str, None, None]:
    """Stream an answer with source citations."""
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        yield "No documents indexed yet. Please upload a PDF first."
        return

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}]\n{c['text']}" for c in chunks
    )
    sources = list({c["source"] for c in chunks})

    system = (
        "You are a helpful assistant that answers questions based strictly on the "
        "provided context. If the context doesn't contain the answer, say so clearly. "
        "Always cite sources.\n\n"
        f"CONTEXT:\n{context}"
    )

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        yield full_response

    yield full_response + f"\n\n---\n**Sources:** {', '.join(sources)}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

def upload_handler(file) -> str:
    if file is None:
        return "No file uploaded."
    n = ingest_pdf(file.name)
    return f"Indexed {n} chunks from {Path(file.name).name}"


with gr.Blocks(title="Document Q&A Bot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Document Q&A Bot\nUpload PDFs and ask questions with source citations.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Index Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)
            upload_btn.click(upload_handler, inputs=file_input, outputs=upload_status)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            question_input = gr.Textbox(placeholder="Ask a question about your documents...", label="Question")
            ask_btn = gr.Button("Ask", variant="primary")

            def chat(question, history):
                history = history or []
                history.append((question, ""))
                for response in answer_stream(question):
                    history[-1] = (question, response)
                    yield history, ""

            ask_btn.click(chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
            question_input.submit(chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])

if __name__ == "__main__":
    demo.launch()
