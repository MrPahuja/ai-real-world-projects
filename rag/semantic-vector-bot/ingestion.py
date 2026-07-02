from pathlib import Path
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP
import database

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
    """Ingest a PDF file into ChromaDB. ChromaDB auto-embeds using local ONNX model."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full_text)
    filename = Path(pdf_path).name

    database.collection.add(
        documents=chunks,
        ids=[f"{filename}::chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)
