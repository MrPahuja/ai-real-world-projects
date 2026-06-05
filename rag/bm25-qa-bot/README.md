# BM25 Document Q&A Bot — Vector-less RAG

A RAG (Retrieval-Augmented Generation) system that answers questions about your PDFs
using **BM25 keyword search** — no vector database, no embeddings API, no GPU required.

## How it works

```
PDF → parse → chunk → BM25Okapi index (in-memory)
                              ↓
            question → tokenise → BM25 rank → top-K chunks
                                                    ↓
                              Gemini Flash (free) → streamed answer
```

| Component | Technology |
|---|---|
| PDF parsing | `pypdf` |
| Retrieval | `rank_bm25` (BM25Okapi) — keyword-based, **no embeddings** |
| Index storage | In-memory Python list — **no vector DB** |
| Answer generation | Google Gemini Flash (free tier) |
| UI | Gradio |

## BM25 vs Vector DB

| | BM25 (this project) | ChromaDB / Qdrant |
|---|---|---|
| Embeddings needed | No | Yes |
| GPU / API cost | None | API cost or local GPU |
| Startup time | Instant | Seconds (model load) |
| Best for | Keyword-rich queries, exact terms | Semantic / paraphrase queries |
| Persistence | Rebuild on restart (or save JSON) | Disk / server |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your GOOGLE_API_KEY
python main.py
```

Open **http://localhost:7860** in your browser.

Get a free Gemini API key at https://aistudio.google.com/apikey

## Usage tips

- **BM25 is keyword-based** — use specific terms, names, or phrases from the document.
- Upload multiple PDFs; the index accumulates all chunks.
- Adjust the **Top-K** slider to retrieve more or fewer chunks as context.
- Click **Clear** to wipe the index and start fresh.
