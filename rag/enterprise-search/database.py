from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import VECTOR_DIM

@dataclass
class DocChunk:
    id: str
    text: str
    source: str
    tenant_id: str
    allowed_users: list   # empty list = every user in the tenant can read it

# In-memory database clients
qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name="kb",
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

# All indexed chunks keyed by chunk_id
chunk_store: dict[str, DocChunk] = {}

# BM25 index per tenant: {tenant_id -> {"chunks": [...], "index": BM25Okapi}}
bm25_store: dict[str, dict] = {}

# Preloaded sample documents
SAMPLE_DOCS = [
    {
        "text": (
            "My Private Journal - June 2024\n\n"
            "June 10: Thinking of getting Mom a nice gardening set for her birthday next month. Need to keep it secret.\n"
            "June 14: Went to the dentist today. Need to pay the remaining invoice of $85.00 by next week.\n"
            "June 18: Started learning Python. It's really fun! The Agentic AI lessons make so much sense."
        ),
        "source": "My Personal Journal",
        "tenant_id": "my-cabinet",
        "allowed_users": ["employee"],
    },
    {
        "text": (
            "Biology Class Group Project Notes\n"
            "Topic: Photosynthesis and Cellular Respiration.\n"
            "Group members: Self, Study Partner.\n\n"
            "Key Details:\n"
            "  1. Photosynthesis converts light energy into chemical energy stored in glucose.\n"
            "  2. Cellular respiration breaks down glucose to produce ATP (energy).\n"
            "  3. Presentation date: July 12. We need to submit the slides by July 10.\n\n"
            "Tasks:\n"
            "  - Self: Research Light-Independent Reactions (Calvin Cycle).\n"
            "  - Study Partner: Design the PowerPoint slides and write the abstract."
        ),
        "source": "Biology Group Project",
        "tenant_id": "my-cabinet",
        "allowed_users": ["employee", "admin"],
    },
    {
        "text": (
            "Alice's Adventures in Wonderland - Public Summary\n"
            "Written by Lewis Carroll in 1865.\n\n"
            "Key Characters:\n"
            "  - Alice: A young girl who falls down a rabbit hole into a fantasy world.\n"
            "  - The White Rabbit: The prompt and anxious rabbit who leads Alice down the hole.\n"
            "  - The Cheshire Cat: A grinning cat who can disappear and reappear at will.\n"
            "  - The Queen of Hearts: The hot-tempered ruler who frequently orders executions ('Off with their heads!').\n\n"
            "Famous Scenes:\n"
            "  1. The Mad Tea-Party with the Mad Hatter and the March Hare.\n"
            "  2. The caucus-race and Alice swimming in a pool of her own tears."
        ),
        "source": "Alice in Wonderland Summary",
        "tenant_id": "my-cabinet",
        "allowed_users": [],  # empty = public (all users)
    },
    {
        "text": (
            "System Admin Configuration Manual - Confidential\n\n"
            "Server IP Configurations:\n"
            "  - Production host: 10.0.1.50 (Port 80/443)\n"
            "  - Staging host:    10.0.1.51 (Port 80/443)\n"
            "  - Vault backup:    10.0.9.12 (Port 8080)\n\n"
            "Emergency contact: sysops@company.com.\n"
            "All database migrations must be approved by the lead architect."
        ),
        "source": "Admin System Manual",
        "tenant_id": "my-cabinet",
        "allowed_users": ["admin"],
    },
]

def corpus_stats(tenant_id: str) -> str:
    """Utility to print statistics of indexed documents in Gradio HTML block."""
    chunks  = [c for c in chunk_store.values() if c.tenant_id == tenant_id]
    sources = sorted({c.source for c in chunks})
    if not chunks:
        return """
        <div style='background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 14px; text-align: center; color: #64748b;'>
            <p style='margin: 0; font-size: 13px;'>No documents indexed yet.</p>
        </div>
        """
    
    sources_li = "".join([f"<li style='margin-bottom: 6px; color: #334155; font-size: 12.5px; display: flex; align-items: center; gap: 6px;'><span>📁</span> <b>{s}</b></li>" for s in sources])
    
    return f"""
    <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; box-shadow: inset 0 1px 2px 0 rgba(0, 0, 0, 0.05);'>
        <p style='margin: 0 0 10px 0; font-size: 13.5px; font-weight: 600; color: #0f172a;'>
            📊 {len(chunks)} chunks across {len(sources)} source(s):
        </p>
        <ul style='margin: 0; padding: 0; list-style-type: none;'>
            {sources_li}
        </ul>
    </div>
    """
