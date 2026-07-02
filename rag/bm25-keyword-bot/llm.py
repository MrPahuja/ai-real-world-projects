import re
import time
from typing import Generator
from google.genai import types

from config import gemini, CHAT_MODEL
import database
from retrieval import retrieve

def answer_stream(question: str, top_k: int = 5) -> Generator[str, None, None]:
    """Stream a Gemini answer grounded in BM25-retrieved context."""
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        if database.bm25 is None:
            yield "No documents have been indexed yet. Please upload and index a PDF first."
        else:
            yield (
                "No relevant matches found in the document.\n\n"
                "BM25 search requires exact keyword matches. Try rephrasing with key terms from the document."
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
            is_transient = "429" in msg or "503" in msg or "UNAVAILABLE" in msg or "experiencing high demand" in msg
            if attempt < 4 and is_transient:
                err_type = "Rate limit hit" if "429" in msg else "Service busy (503)"
                for remaining in range(wait, 0, -1):
                    yield f"[ {err_type} - retrying in {remaining}s... ]"
                    time.sleep(1)
                full_response = ""
            else:
                if "429" in msg:
                    yield "[ Rate limit exceeded. Please wait a minute and try again. ]"
                elif "503" in msg or "UNAVAILABLE" in msg:
                    yield "[ Service temporarily unavailable. Please try again shortly. ]"
                else:
                    yield f"[ Error: {msg[:300]} ]"
                return

    yield full_response + f"\n\n---\n**Sources:** {', '.join(sources)}"
