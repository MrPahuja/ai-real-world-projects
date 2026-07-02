import os
import gradio as gr

from config import TENANT, USERS, USER_ACCESS_GUIDE, CHAT_MODEL, EMBED_MODEL
from database import corpus_stats, SAMPLE_DOCS
from ingestion import upload_pdf, index_document
from llm import respond

# ── CSS styling for Gradio UI ──────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #f1f5f9 !important;
}
.header-container {
    background: white;
    border-radius: 14px;
    padding: 8px 16px;
    margin-bottom: 6px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 6px rgba(0,0,0,.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.header-title-section h1 { margin: 0 0 4px 0; font-size: 22px; font-weight: 700; color: #0f172a; }
.header-title-section p  { margin: 0; font-size: 13px; color: #64748b; }
.badge-container { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.badge {
    padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: .3px;
    background: #fdf4ff; color: #7c3aed; border: 1px solid #e9d5ff;
}
.badge.local  { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }
.badge.api    { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.panel-block {
    background: white; border-radius: 12px; padding: 16px;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.panel-label {
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 1.2px !important; color: #64748b !important;
    text-transform: uppercase !important; margin: 16px 0 6px 0 !important;
}
.panel-label:first-child { margin-top: 0 !important; }
.access-guide {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #475569; line-height: 1.8;
}
.access-guide b { color: #0f172a; }
.access-guide .tip {
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid #e2e8f0; font-style: italic; color: #94a3b8;
}
.download-btn-custom {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    text-align: left !important;
    color: #1e293b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-bottom: 8px !important;
    box-shadow: none !important;
    width: 100% !important;
    min-height: 42px !important;
}
.download-btn-custom:hover {
    border-color: #ea580c !important;
    background: #fff7ed !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    color: #ea580c !important;
}
"""

def generate_sample_pdfs():
    """Builds standard PDF files of the sample data using ReportLab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    os.makedirs("sample_files", exist_ok=True)
    styles = getSampleStyleSheet()
    
    # 1. My Personal Journal PDF
    pdf_path1 = "sample_files/My_Personal_Journal.pdf"
    if not os.path.exists(pdf_path1):
        doc = SimpleDocTemplate(pdf_path1, pagesize=letter)
        story = [
            Paragraph("<b>My Private Journal - June 2024</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph("<b>June 10:</b> Thinking of getting Mom a nice gardening set for her birthday next month. Need to keep it secret.", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>June 14:</b> Went to the dentist today. Need to pay the remaining invoice of $85.00 by next week.", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>June 18:</b> Started learning Python. It's really fun! The Agentic AI lessons make so much sense.", styles["Normal"]),
        ]
        doc.build(story)

    # 2. Biology Group Project PDF
    pdf_path2 = "sample_files/Biology_Group_Project.pdf"
    if not os.path.exists(pdf_path2):
        doc = SimpleDocTemplate(pdf_path2, pagesize=letter)
        story = [
            Paragraph("<b>Biology Class Group Project Notes</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph("<b>Topic:</b> Photosynthesis and Cellular Respiration.<br/><b>Group members:</b> Self, Study Partner.", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Key Details:</b>", styles["Heading3"]),
            Paragraph("1. Photosynthesis converts light energy into chemical energy stored in glucose.", styles["Normal"]),
            Paragraph("2. Cellular respiration breaks down glucose to produce ATP (energy).", styles["Normal"]),
            Paragraph("3. Presentation date: July 12. We need to submit the slides by July 10.", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>Tasks:</b>", styles["Heading3"]),
            Paragraph("- Self: Research Light-Independent Reactions (Calvin Cycle).", styles["Normal"]),
            Paragraph("- Study Partner: Design the PowerPoint slides and write the abstract.", styles["Normal"]),
        ]
        doc.build(story)

    # 3. Alice in Wonderland Summary PDF
    pdf_path3 = "sample_files/Alice_in_Wonderland_Summary.pdf"
    if not os.path.exists(pdf_path3):
        doc = SimpleDocTemplate(pdf_path3, pagesize=letter)
        story = [
            Paragraph("<b>Alice's Adventures in Wonderland - Public Summary</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph("Written by Lewis Carroll in 1865.", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Key Characters:</b>", styles["Heading3"]),
            Paragraph("- <b>Alice:</b> A young girl who falls down a rabbit hole into a fantasy world.", styles["Normal"]),
            Paragraph("- <b>The White Rabbit:</b> The prompt and anxious rabbit who leads Alice down the hole.", styles["Normal"]),
            Paragraph("- <b>The Cheshire Cat:</b> A grinning cat who can disappear and reappear at will.", styles["Normal"]),
            Paragraph("- <b>The Queen of Hearts:</b> The hot-tempered ruler who frequently orders executions ('Off with their heads!').", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>Famous Scenes:</b>", styles["Heading3"]),
            Paragraph("1. The Mad Tea-Party with the Mad Hatter and the March Hare.", styles["Normal"]),
            Paragraph("2. The caucus-race and Alice swimming in a pool of her own tears.", styles["Normal"]),
        ]
        doc.build(story)

    # 4. Admin System Manual PDF
    pdf_path4 = "sample_files/Admin_System_Manual.pdf"
    if not os.path.exists(pdf_path4):
        doc = SimpleDocTemplate(pdf_path4, pagesize=letter)
        story = [
            Paragraph("<b>System Admin Configuration Manual - Confidential</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph("<b>Server IP Configurations:</b>", styles["Heading3"]),
            Paragraph("- Production host: 10.0.1.50 (Port 80/443)", styles["Normal"]),
            Paragraph("- Staging host:    10.0.1.51 (Port 80/443)", styles["Normal"]),
            Paragraph("- Vault backup:    10.0.9.12 (Port 8080)", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("<b>Security & Administration Guidelines:</b>", styles["Heading3"]),
            Paragraph("Emergency contact: sysops@company.com.<br/>All database migrations must be approved by the lead architect.", styles["Normal"]),
        ]
        doc.build(story)


def load_samples():
    """Generates physical PDFs and indexes sample documents at startup."""
    generate_sample_pdfs()
    for doc in SAMPLE_DOCS:
        index_document(doc["text"], doc["source"], doc["tenant_id"], doc["allowed_users"])


# ── Gradio blocks interface ────────────────────────────────────────────────────
def build_interface():

    with gr.Blocks(title="Smart File Cabinet") as demo:
        tenant_state = gr.State(TENANT)

        gr.HTML(f"""
        <div class="header-container">
            <div class="header-title-section">
                <h1>Smart File Cabinet</h1>
                <p>Hybrid retrieval (BM25 + vector + RRF) with user-level access control.
                   Securely chat with your personal diaries, shared class project notes, and public reference files.</p>
            </div>
            <div class="badge-container">
                <span class="badge">Retrieval: BM25 + Vector + RRF</span>
                <span class="badge local">Embeddings: gemini-embedding-001 (free)</span>
                <span class="badge api">LLM: Gemini 2.5 Flash</span>
                <span class="badge local">Access: User-Level ACL</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            # LEFT PANEL (Vault & Uploads)
            with gr.Column(scale=1, min_width=290, elem_classes=["panel-block"]):
                with gr.Tabs():
                    with gr.Tab("📁 Vault Corpus"):
                        gr.HTML('<p class="panel-label">Active Corpus</p>')
                        stats_box = gr.HTML(
                            value=corpus_stats(TENANT),
                        )
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=3):
                                gr.HTML("""
                                <div style="font-size: 11px; line-height: 1.3; color: #475569; padding-top: 2px;">
                                    💡 These 4 documents are pre-loaded in your cabinet. Download their physical PDFs below to inspect.
                                </div>
                                """)
                            with gr.Column(scale=1, min_width=75):
                                refresh_btn = gr.Button("Refresh", size="sm")
                        
                        gr.HTML('<p class="panel-label" style="margin-top: 10px !important;">Download Sample PDFs</p>')
                        gr.DownloadButton(
                            label="📄 My Personal Journal (PDF)",
                            value=os.path.abspath("sample_files/My_Personal_Journal.pdf"),
                            elem_classes=["download-btn-custom"]
                        )
                        gr.DownloadButton(
                            label="📄 Biology Group Project Notes (PDF)",
                            value=os.path.abspath("sample_files/Biology_Group_Project.pdf"),
                            elem_classes=["download-btn-custom"]
                        )
                        gr.DownloadButton(
                            label="📄 Alice in Wonderland Summary (PDF)",
                            value=os.path.abspath("sample_files/Alice_in_Wonderland_Summary.pdf"),
                            elem_classes=["download-btn-custom"]
                        )
                        gr.DownloadButton(
                            label="📄 Admin System Manual (PDF)",
                            value=os.path.abspath("sample_files/Admin_System_Manual.pdf"),
                            elem_classes=["download-btn-custom"]
                        )

                    with gr.Tab("📤 Upload New PDF"):
                        gr.HTML("""
                        <div style="margin-bottom: 8px; font-size: 0.8em; line-height: 1.3; color: #475569; background-color: #f8fafc; padding: 6px; border: 1px solid #e2e8f0; border-radius: 4px;">
                            <b>Need test files?</b> Download online PDFs:
                            <ul style="margin: 2px 0 0 12px; padding: 0;">
                                <li><a href="https://bitcoin.org/bitcoin.pdf" target="_blank" style="color: #ea580c; text-decoration: underline;">Bitcoin Whitepaper</a></li>
                                <li><a href="https://www.gutenberg.org/files/1661/1661-pdf.pdf" target="_blank" style="color: #ea580c; text-decoration: underline;">Sherlock Holmes</a></li>
                            </ul>
                        </div>
                        """)
                        pdf_file   = gr.File(label="PDF file", file_types=[".pdf"])
                        source_in  = gr.Textbox(label="Document name", placeholder="e.g. Textbook")
                        users_in   = gr.Textbox(
                            label="Allowed users (comma-separated)",
                            placeholder="self, admin",
                        )
                        upload_btn = gr.Button("Index document", variant="primary", size="sm")
                        upload_out = gr.Textbox(label="Status", interactive=False, lines=2)

                # Wiring Left Column Actions
                refresh_btn.click(fn=lambda: corpus_stats(TENANT), outputs=[stats_box])
                upload_btn.click(
                    fn=upload_pdf,
                    inputs=[pdf_file, source_in, tenant_state, users_in],
                    outputs=[upload_out],
                )

            # RIGHT PANEL: MAIN CHAT & DETAILS
            with gr.Column(scale=3):
                # Compact User Selector and Guide inline above Chatbot
                with gr.Row(variant="compact"):
                    with gr.Column(scale=1, min_width=90):
                        gr.HTML("""
                        <div style="font-size: 12px; font-weight: 600; color: #475569; margin-top: 8px; text-align: right;">
                            👤 Active User:
                        </div>
                        """)
                    with gr.Column(scale=2, min_width=120):
                        user_dd = gr.Dropdown(
                            choices=USERS, value="guest", 
                            show_label=False, 
                            interactive=True
                        )
                    with gr.Column(scale=6):
                        gr.HTML("""
                        <div style="font-size: 10.5px; line-height: 1.3; color: #475569; padding: 6px 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;">
                            🔑 <b>Access Matrix:</b> <b>employee</b> (Journal, Biology, Alice) | <b>admin</b> (All files + Admin Manual) | <b>guest</b> (Alice only)
                        </div>
                        """)

                with gr.Tabs():
                    with gr.Tab("💬 Cabinet Chatbot"):
                        chatbot = gr.Chatbot(
                            height=280,
                            label="",
                            show_label=False,
                            render_markdown=True,
                        )
                        with gr.Row():
                            msg_in   = gr.Textbox(
                                placeholder="Ask anything about your documents...",
                                show_label=False, scale=5, lines=1,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                        clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

                    with gr.Tab("🔍 Retrieval Logs & Metrics"):
                        debug_output = gr.Markdown(
                            value="*Ask a question to see the step-by-step mathematical retrieval details (BM25 vs. Vector scores).* "
                        )

                # Wiring Right Column Actions
                clear_btn.click(
                    fn=lambda: ([], "*Ask a question to see the step-by-step mathematical retrieval details (BM25 vs. Vector scores).* "), 
                    outputs=[chatbot, debug_output]
                )

                def _submit(msg, hist, tenant, user):
                    yield from respond(msg, hist, tenant, user)

                send_btn.click(
                    fn=_submit,
                    inputs=[msg_in, chatbot, tenant_state, user_dd],
                    outputs=[msg_in, chatbot, debug_output],
                )
                msg_in.submit(
                    fn=_submit,
                    inputs=[msg_in, chatbot, tenant_state, user_dd],
                    outputs=[msg_in, chatbot, debug_output],
                )
                
    return demo
