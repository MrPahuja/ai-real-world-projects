"""
BM25 Document Q&A Bot (Vector-less RAG)
========================================
Ingests PDFs, indexes them with BM25 keyword search (no embeddings,
no vector database), and answers questions using Google Gemini Flash.

How it works:
  1. PDFs are parsed and split into overlapping text chunks.
  2. Chunks are tokenised and indexed in a BM25Okapi index (in-memory).
  3. At query time, BM25 ranks chunks by keyword relevance — no vectors needed.
  4. The top-K chunks form the context sent to Gemini for answer generation.

No vector DB. No embeddings API. No GPU. Runs entirely on CPU.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # add GOOGLE_API_KEY
    python main.py

UI launches at http://localhost:7860
Get a free API key at: https://aistudio.google.com/apikey
"""

import os
import re
import string
import time
from pathlib import Path
from typing import Generator

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()

gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

CHAT_MODEL = "gemini-flash-latest"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# ── In-memory index (no vector DB) ───────────────────────────────────────────

# Each entry: {"text": str, "source": str, "chunk": int}
_corpus: list[dict] = []
_bm25: BM25Okapi | None = None


def _tokenise(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace — classic BM25 tokens."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _rebuild_index() -> None:
    """Rebuild the BM25 index from the current corpus."""
    global _bm25
    if _corpus:
        _bm25 = BM25Okapi([_tokenise(doc["text"]) for doc in _corpus])
    else:
        _bm25 = None


# ── Ingestion ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(pdf_path: str) -> int:
    """Parse PDF, chunk text, append to corpus, rebuild BM25 index."""
    global _corpus
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full_text)
    filename = Path(pdf_path).name

    _corpus.extend(
        {"text": chunk, "source": filename, "chunk": i}
        for i, chunk in enumerate(chunks)
    )
    _rebuild_index()
    return len(chunks)


def clear_index() -> None:
    """Wipe the in-memory corpus and index."""
    global _corpus, _bm25
    _corpus = []
    _bm25 = None


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Return top-K chunks ranked by BM25 score."""
    if _bm25 is None or not _corpus:
        return []
    tokens = _tokenise(question)
    scores = _bm25.get_scores(tokens)
    ranked = sorted(
        zip(scores, _corpus), key=lambda x: x[0], reverse=True
    )
    return [
        {**doc, "score": float(score)}
        for score, doc in ranked[:top_k]
        if score > 0
    ]


# ── Generation ───────────────────────────────────────────────────────────────

def answer_stream(question: str, top_k: int = 5) -> Generator[str, None, None]:
    """Stream a Gemini answer grounded in BM25-retrieved context."""
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        if _bm25 is None:
            yield "📂 No documents indexed yet. Please upload and index a PDF first."
        else:
            yield (
                "🔍 No relevant chunks found for your query.\n\n"
                "BM25 is keyword-based — try rephrasing with key terms from the document."
            )
        return

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}, BM25={c['score']:.2f}]\n{c['text']}"
        for c in chunks
    )
    sources = list({c["source"] for c in chunks})

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the "
        "provided context. If the context doesn't contain the answer, say so clearly. "
        "Always cite the source document.\n\n"
        f"CONTEXT:\n{context}"
    )

    full_response = ""
    for attempt in range(5):
        try:
            for chunk in gemini.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=question,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            ):
                if chunk.text:
                    full_response += chunk.text
                    yield full_response
            break
        except Exception as e:
            msg = str(e)
            delay_match = re.search(r"retryDelay.*?(\d+)s", msg)
            wait = int(delay_match.group(1)) + 2 if delay_match else 30
            if attempt < 4 and "429" in msg:
                for remaining in range(wait, 0, -1):
                    yield f"⏳ **Rate limit hit** — retrying in **{remaining}s**..."
                    time.sleep(1)
                full_response = ""
            else:
                yield (
                    "❌ **Rate limit exhausted**. Wait a minute and try again."
                    if "429" in msg
                    else f"❌ **Error** — {msg[:300]}"
                )
                return

    yield full_response + f"\n\n---\n📎 **Sources:** {', '.join(sources)}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
.title-block { text-align: center; padding: 16px 0 8px; }
.title-block h1 { font-size: 2rem; margin-bottom: 4px; }
.title-block p  { color: #6b7280; font-size: 0.95rem; }
.badge { display: inline-block; background: #f3f4f6; border-radius: 6px;
         padding: 2px 10px; font-size: 0.8rem; color: #374151; margin: 2px; }
.panel-label { font-weight: 600; font-size: 1rem; margin-bottom: 4px; color: #111827; }
.tip-box { background: #fefce8; border: 1px solid #fde047; border-radius: 8px;
           padding: 10px 14px; font-size: 0.85rem; color: #713f12; margin-top: 8px; }
footer { display: none !important; }
"""


def upload_handler(file):
    if file is None:
        return "⚠️ No file selected. Please choose a PDF."
    try:
        n = ingest_pdf(file.name)
        total = len(_corpus)
        return f"✅ Indexed **{n} new chunks** from `{Path(file.name).name}` — total corpus: **{total} chunks**"
    except Exception as e:
        return f"❌ Indexing failed: {str(e)[:200]}"


def clear_handler():
    clear_index()
    return "🗑️ Index cleared. Upload a new PDF to begin."


def corpus_stats() -> str:
    docs = list({c["source"] for c in _corpus})
    if not docs:
        return "No documents indexed."
    return f"**{len(_corpus)} chunks** across **{len(docs)} document(s):** " + ", ".join(f"`{d}`" for d in docs)


with gr.Blocks(title="BM25 Q&A Bot", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="title-block">
        <h1>📄 BM25 Document Q&A Bot</h1>
        <p>Vector-less RAG — keyword search with BM25, no embeddings, no vector database.</p>
        <span class="badge">🔍 Retrieval: BM25Okapi (rank-bm25)</span>
        <span class="badge">✨ Chat: Gemini Flash (free)</span>
        <span class="badge">⚡ No vector DB · No embeddings API</span>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=300):
            gr.HTML('<p class="panel-label">📁 Document</p>')
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            with gr.Row():
                upload_btn = gr.Button("⚡ Index Document", variant="primary", scale=3)
                clear_btn  = gr.Button("🗑️ Clear", variant="secondary", scale=1)
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                placeholder="Upload a PDF and click Index Document...",
            )
            stats_md = gr.Markdown("No documents indexed.")

            gr.HTML("""
            <div class="tip-box">
              💡 <strong>BM25 tip:</strong> BM25 is keyword-based — it works best
              when your question shares vocabulary with the document.
              Use specific terms, names, or phrases from the PDF.
            </div>
            """)

            upload_btn.click(upload_handler, inputs=file_input, outputs=upload_status)
            upload_btn.click(corpus_stats, outputs=stats_md)
            clear_btn.click(clear_handler, outputs=upload_status)
            clear_btn.click(corpus_stats, outputs=stats_md)

        with gr.Column(scale=2):
            gr.HTML('<p class="panel-label">💬 Chat</p>')
            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                placeholder="Index a document on the left, then ask a question below.",
                avatar_images=(
                    None,
                    "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg",
                ),
            )
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                ask_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)
            with gr.Row():
                clear_chat_btn = gr.Button("🗑️ Clear chat", variant="secondary", size="sm")
                top_k_slider = gr.Slider(
                    minimum=1, maximum=15, value=5, step=1,
                    label="Top-K chunks", scale=2,
                )

            def chat(question, history, top_k):
                if not question.strip():
                    yield history, ""
                    return
                history = history or []
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": ""})
                for response in answer_stream(question, top_k=int(top_k)):
                    history[-1] = {"role": "assistant", "content": response}
                    yield history, ""

            ask_btn.click(
                chat,
                inputs=[question_input, chatbot, top_k_slider],
                outputs=[chatbot, question_input],
            )
            question_input.submit(
                chat,
                inputs=[question_input, chatbot, top_k_slider],
                outputs=[chatbot, question_input],
            )
            clear_chat_btn.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
