from pathlib import Path
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP
import database

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def ingest_pdf(pdf_path: str) -> int:
    """Parse PDF, chunk text, append to corpus, rebuild BM25 index."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full_text)
    filename = Path(pdf_path).name

    database.corpus.extend(
        {"text": chunk, "source": filename, "chunk": i}
        for i, chunk in enumerate(chunks)
    )
    database.rebuild_index()
    return len(chunks)
