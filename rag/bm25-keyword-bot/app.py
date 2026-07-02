from pathlib import Path
import gradio as gr

import database
from ingestion import ingest_pdf
from llm import answer_stream

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Quicksand:wght@600;700;800&display=swap');

body, .gradio-container, button, input, select, textarea, span, p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

h1, h2, h3, .panel-label, .title-block h1 {
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 700 !important;
}

.gradio-container {
    background-color: #fafbfc !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.05) 0, transparent 40%),
        radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.05) 0, transparent 40%),
        radial-gradient(at 50% 100%, rgba(244, 63, 94, 0.04) 0, transparent 50%) !important;
    color: #1e293b !important;
    max-width: 1200px !important;
    padding: 12px 24px !important;
}

.panel-block {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px -5px rgba(100, 116, 139, 0.08) !important;
    padding: 20px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
.panel-block:hover {
    border-color: #6366f1 !important;
    box-shadow: 0 16px 35px -8px rgba(99, 102, 241, 0.12) !important;
    transform: translateY(-1px) !important;
}

.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px -4px rgba(100, 116, 139, 0.05);
}
.header-title-section {
    text-align: left;
}
.header-title-section h1 {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0 0 2px 0;
    background: linear-gradient(90deg, #6366f1 0%, #3b82f6 35%, #ec4899 70%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.header-title-section p {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0;
}

.badge-container {
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 8px;
}
.badge {
    display: inline-block;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    border-radius: 9999px !important;
    padding: 4px 14px !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease !important;
}
.badge:hover {
    background: rgba(99, 102, 241, 0.08) !important;
    border-color: #6366f1 !important;
    color: #6366f1 !important;
}

.panel-label {
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 12px;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

input, textarea, .file-preview, .file-input, .gr-file {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.2s ease !important;
}
input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

button.primary, button.lg.primary {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
button.primary:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
    filter: brightness(1.05) !important;
}
button.primary:active {
    transform: scale(0.98) !important;
}

button.secondary, button.lg.secondary {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    color: #475569 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
button.secondary:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #1e293b !important;
}

.chatbot-container {
    border-radius: 16px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
}
.message.user {
    background: #e0e7ff !important;
    border: 1px solid #c7d2fe !important;
    color: #1e1b4b !important;
    border-radius: 12px 12px 0px 12px !important;
}
.message.bot {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 4px solid #3b82f6 !important;
    color: #0f172a !important;
    border-radius: 12px 12px 12px 0px !important;
}

.slider-container input[type="range"] {
    accent-color: #3b82f6 !important;
}

.tip-box {
    background: #fefce8 !important;
    border: 1px solid #fef08a !important;
    border-radius: 16px !important;
    padding: 16px !important;
    color: #713f12 !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}
.tip-box strong {
    color: #a16207 !important;
    font-weight: 700;
    display: block;
    margin-bottom: 4px;
}

textarea[readonly] {
    background: #f8fafc !important;
    border-color: #e2e8f0 !important;
    color: #64748b !important;
}

footer { display: none !important; }
"""

def upload_handler(file):
    if file is None:
        return "No file selected. Please choose a PDF."
    try:
        n = ingest_pdf(file.name)
        total = len(database.corpus)
        return f"Successfully indexed {n} chunks from {Path(file.name).name}. Total corpus size: {total} chunks."
    except Exception as e:
        return f"Indexing failed: {str(e)[:200]}"

def clear_handler():
    database.clear_index()
    return "Index successfully cleared. Ready for a new PDF document."

def corpus_stats() -> str:
    docs = list({c["source"] for c in database.corpus})
    if not docs:
        return "No documents indexed."
    return f"{len(database.corpus)} chunks across {len(docs)} document(s): " + ", ".join(f"`{d}`" for d in docs)

def build_interface():
    with gr.Blocks(title="BM25 Keyword Bot", css=CSS) as demo:
        gr.HTML("""
        <div class="header-container">
            <div class="header-title-section">
                <h1>BM25 Keyword Bot</h1>
                <p>Keyword-powered document search. No embeddings, no vector database. Just fast, local BM25 retrieval and Gemini answers.</p>
            </div>
            <div class="badge-container">
                <span class="badge">Retrieval: BM25 (rank-bm25)</span>
                <span class="badge">Language Model: Gemini Flash</span>
                <span class="badge">Architecture: Vector-less &amp; Local</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300, elem_classes=["panel-block"]):
                gr.HTML('<p class="panel-label">Document Ingestion</p>')
                file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                with gr.Row():
                    upload_btn = gr.Button("Index Document", variant="primary", scale=3)
                    clear_btn  = gr.Button("Clear Index", variant="secondary", scale=1)
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                    placeholder="Upload a PDF and click Index Document...",
                )
                stats_md = gr.Markdown("No documents indexed.")
                
                top_k_slider = gr.Slider(
                    minimum=1, maximum=15, value=5, step=1,
                    label="Top-K chunks", scale=1,
                )

                gr.HTML("""
                <div class="tip-box">
                  <strong>BM25 Search Tip:</strong> BM25 is keyword-based. It matches exact terms between your query and the document. For best results, use specific words, names, or phrases from the PDF.
                </div>
                """)

                upload_btn.click(upload_handler, inputs=file_input, outputs=upload_status)
                upload_btn.click(corpus_stats, outputs=stats_md)
                clear_btn.click(clear_handler, outputs=upload_status)
                clear_btn.click(corpus_stats, outputs=stats_md)

            with gr.Column(scale=2, elem_classes=["panel-block"]):
                gr.HTML('<p class="panel-label">Assistant Console</p>')
                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                    type="messages",
                    placeholder="Upload and index a document on the left, then ask a question here.",
                    elem_classes=["chatbot-container"],
                )
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Ask a question about the document...",
                        show_label=False,
                        scale=4,
                        container=False,
                    )
                    ask_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
                    clear_chat_btn = gr.Button("Clear Chat", variant="secondary", scale=1, min_width=80)

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

    return demo
