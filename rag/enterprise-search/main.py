"""
Enterprise Search Entrypoint
===========================
Production-grade Multi-Tenant RAG system with hybrid search, RRF re-ranking, 
user access control lists, and scanned PDF OCR fallback capabilities.
"""

from app import load_samples, build_interface, CSS

if __name__ == "__main__":
    # Initialize and index sample documents
    load_samples()
    
    # Build Gradio Block Interface and launch the web UI
    demo = build_interface()
    import os
    demo.launch(css=CSS, allowed_paths=[os.path.abspath("sample_files")])
