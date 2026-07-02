import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Always configure Gemini for embeddings & OCR fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Config constants
EMBED_MODEL   = "models/gemini-embedding-001"   # free, 768/3072 dims
PROVIDER      = os.getenv("LLM_PROVIDER", "gemini").lower()
CHAT_MODEL    = os.getenv("CHAT_MODEL", "gemini-1.5-flash-latest")
VECTOR_DIM    = 3072
CHUNK_SIZE    = 900     # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
BM25_POOL     = 20      # candidates from BM25 before ACL filter
VECTOR_POOL   = 20      # candidates from vector search before ACL filter
FINAL_TOP_K   = 5       # chunks passed to the LLM
RRF_K         = 60      # RRF constant (standard value)

# Tenants and access levels
TENANT = "my-cabinet"
USER_ACCESS_GUIDE = {
    "employee": "Personal journal, Biology project, and public reference files",
    "admin":    "All files (Admin Configuration Manual + employee/public files)",
    "guest":    "Public reference files only (Alice in Wonderland)",
}
USERS = ["guest", "employee", "admin"]

# Initialize OpenAI/OpenRouter client if needed
openai_client = None
if PROVIDER == "openai":
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
elif PROVIDER == "openrouter":
    from openai import OpenAI
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        default_headers={
            "HTTP-Referer": "https://github.com/enterprise-search-bot",
            "X-Title": "Enterprise Search Bot"
        }
    )
