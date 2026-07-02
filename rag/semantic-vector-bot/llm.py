import re
import time
from typing import Generator
from google.genai import types

from config import gemini, CHAT_MODEL
from retrieval import retrieve

def answer_stream(question: str, top_k: int = 5) -> Generator[str, None, None]:
    """Stream an answer with source citations and auto-retry on rate limit."""
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        yield "📂 No documents indexed yet. Please upload and index a PDF first."
        return

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}]\n{c['text']}" for c in chunks
    )
    sources = list({c["source"] for c in chunks})

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the "
        "provided context. If the context doesn't contain the answer, say so clearly. "
        "Always cite sources.\n\n"
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
                    yield f"⏳ **Rate limit hit** — free tier quota reached. Retrying in **{remaining}s**..."
                    time.sleep(1)
                full_response = ""
            else:
                if "429" in msg:
                    yield "❌ **Rate limit exhausted** — all retry attempts failed. Please wait a minute and try again."
                else:
                    yield f"❌ **Error** — {msg[:300]}"
                return

    yield full_response + f"\n\n---\n📎 **Sources:** {', '.join(sources)}"
