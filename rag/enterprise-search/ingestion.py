import os
import uuid
import string
import pypdf
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from qdrant_client.models import PointStruct

from config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from database import qdrant, chunk_store, bm25_store, DocChunk

def embed(text: str) -> list[float]:
    """Call Google text-embedding-004. Free tier, 768/3072 dimensions."""
    result = genai.embed_content(model=EMBED_MODEL, content=text[:8000])
    return result["embedding"]


def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation before BM25 tokenisation."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def index_document(text: str, source: str, tenant_id: str, allowed_users: list[str]) -> int:
    """
    Chunk, embed, and store a document.
    Rebuilds the BM25 index for the tenant after every upload.
    """
    chunks = chunk_text(text)
    for raw in chunks:
        cid = str(uuid.uuid4())
        emb = embed(raw)
        doc = DocChunk(id=cid, text=raw, source=source,
                       tenant_id=tenant_id, allowed_users=allowed_users)
        chunk_store[cid] = doc

        qdrant.upsert(
            collection_name="kb",
            points=[
                PointStruct(
                    id=cid,
                    vector=emb,
                    payload={
                        "chunk_id": cid,
                        "text": raw,
                        "source": source,
                        "tenant_id": tenant_id,
                        "allowed_users": allowed_users,
                    },
                )
            ],
        )

    # Rebuild BM25 for this tenant so new documents are immediately searchable.
    tenant_chunks = [c for c in chunk_store.values() if c.tenant_id == tenant_id]
    bm25_store[tenant_id] = {
        "chunks": tenant_chunks,
        "index": BM25Okapi([tokenize(c.text) for c in tenant_chunks]),
    }
    return len(chunks)


def ocr_pdf_with_gemini(file_path: str) -> str:
    """Upload a scanned PDF to Google's Files API and extract text using Gemini."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is not set. Cannot run OCR fallback.")
    
    # Upload the file to Gemini Files API
    uploaded_file = genai.upload_file(path=file_path)
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = (
            "Perform OCR on this document. Transcribe all text, tables, and handwritten notes page-by-page. "
            "Preserve the original layout as much as possible, converting tables to Markdown format."
        )
        response = model.generate_content([uploaded_file, prompt])
        return response.text
    finally:
        # Clean up the file from the Google files hosting
        uploaded_file.delete()


def upload_pdf(file, source_name: str, tenant_id: str, users_str: str):
    """Gradio handler to parse uploaded PDF (with OCR fallback) and run indexer."""
    if file is None:
        return "No file selected."
    if not source_name.strip():
        return "Please enter a document name."

    reader  = pypdf.PdfReader(file.name)
    text    = "\n".join(page.extract_text() or "" for page in reader.pages)
    
    status_msg = ""
    # Fallback to Gemini OCR if text is empty or too short (scanned PDF detection)
    if len(text.strip()) < 50:
        try:
            text = ocr_pdf_with_gemini(file.name)
            status_msg = " [OCR Fallback used]"
        except Exception as e:
            return f"Failed to extract text using standard parser, and OCR failed: {str(e)}"

    if not text.strip():
        return "Could not extract text from this PDF."

    allowed = [u.strip() for u in users_str.split(",") if u.strip()] if users_str.strip() else []
    n       = index_document(text, source_name.strip(), tenant_id.strip(), allowed)
    access  = ", ".join(allowed) if allowed else "all users in tenant"
    return f"Indexed {n} chunk(s) from '{source_name}'.{status_msg}\nAccess granted to: {access}."
