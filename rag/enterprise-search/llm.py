import google.generativeai as genai

from config import PROVIDER, CHAT_MODEL, FINAL_TOP_K, openai_client
from database import chunk_store
from retrieval import hybrid_search

def respond(message: str, history: list, tenant_id: str, user_id: str):
    """Stream an answer from Gemini Flash, OpenAI, or OpenRouter using hybrid-retrieved context."""
    if not message.strip():
        yield "", history, ""
        return

    # 1. Immediately append the user message to history and yield
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield "", history, "🔍 *Searching database...*"

    if not chunk_store:
        history[-1]["content"] = "No documents indexed yet. Upload a PDF to get started."
        yield "", history, "⚠️ No documents indexed yet."
        return

    # 2. Run retrieval inside a try-except block
    try:
        chunks, debug = hybrid_search(message, tenant_id, user_id)
    except Exception as e:
        history[-1]["content"] = f"Retrieval Error: {str(e)}"
        yield "", history, f"❌ Retrieval Error: {str(e)}"
        return

    if not chunks:
        history[-1]["content"] = (
            f"No documents found that **{user_id}** has access to for this query.\n\n"
            f"Try switching user roles or uploading a new document."
        )
        yield "", history, "⚠️ No matching chunks found."
        return

    # Construct the RAG prompt context
    context = "\n\n".join(
        f"[Source: {c.source}]\n{c.text}" for c in chunks
    )
    prompt = (
        f"You are a secure, multi-tenant file assistant. You only answer questions using the provided context blocks. "
        f"Do not assume or use external facts unless explicitly allowed. "
        f"If the answer cannot be found in the context blocks, explain that you cannot find the answer and why.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {message}\n"
        f"Answer:"
    )

    # Format debug dashboard output
    debug_md = (
        f"### 🔍 Retrieval Step-by-Step Details\n\n"
        f"1. **Lexical Retrieval (BM25)**\n"
        f"   - Match count: `{debug['bm25_hits']}`\n"
        f"   - Sources matched: `{', '.join(debug['bm25_sources']) or 'None'}`\n\n"
        f"2. **Vector Retrieval (Qdrant Client)**\n"
        f"   - Match count: `{debug['vector_hits']}`\n"
        f"   - Sources matched: `{', '.join(debug['vec_sources']) or 'None'}`\n\n"
        f"🔀 **Reciprocal Rank Fusion (RRF)**\n"
        f"- Merged candidates: `{', '.join(set(debug['final'])) or 'None'}`\n"
        f"- Top **{FINAL_TOP_K}** highest-ranked chunks sent as LLM context."
    )

    # 3. Stream from model
    full = ""
    if PROVIDER in ("openai", "openrouter"):
        if not openai_client:
            history[-1]["content"] = f"Error: Provider '{PROVIDER}' selected, but client is not initialized. Please verify your API keys."
            yield "", history, "❌ Configuration Error"
            return

        try:
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful knowledge assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                if delta:
                    full += delta
                    history[-1]["content"] = full
                    yield "", history, debug_md
        except Exception as e:
            history[-1]["content"] = f"API Error ({PROVIDER}): {str(e)}"
            yield "", history, f"❌ API Error ({PROVIDER})"
            return
    else:
        # Default Gemini flow
        try:
            model    = genai.GenerativeModel(CHAT_MODEL)
            response = model.generate_content(prompt, stream=True)
            for part in response:
                full += part.text
                history[-1]["content"] = full
                yield "", history, debug_md
        except Exception as e:
            history[-1]["content"] = f"API Error (gemini): {str(e)}"
            yield "", history, f"❌ API Error (gemini)"
            return

    history[-1]["content"] = full
    yield "", history, debug_md
