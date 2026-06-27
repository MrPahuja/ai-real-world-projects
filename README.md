# AI Real-World Projects

Hands-on projects that turn every concept from [AI Learning Hub](https://github.com/MrPahuja/ai-learning-hub) into working code.

Each project is self-contained with its own README, setup instructions, and real working code. These are not toy examples or stubs. You can run every one of them.

---

## What is in here

```
ai-real-world-projects/
|
|-- rag/
|   |-- bm25-keyword-bot/          PDF Q&A using keyword search, no embeddings needed
|   |-- semantic-vector-bot/       PDF Q&A using semantic vector search with ChromaDB
|   |-- enterprise-search/     Multi-tenant RAG with access control
|
|-- mcp/
|   |-- github-server/         MCP server that exposes GitHub as tools
|   |-- postgres-server/       MCP server for safe Postgres querying
|
|-- agentic/
|   |-- research-agent/        ReAct agent for autonomous multi-step research
|   |-- code-review-agent/     Multi-agent PR code reviewer integrated with GitHub
```

---

## Projects

### RAG (Retrieval-Augmented Generation)

RAG is the technique of finding relevant pieces of a document and using them as context when asking an AI a question. All three projects below use RAG but with different retrieval methods.

---

#### `rag/bm25-keyword-bot`

Upload a PDF and ask questions about it. The search runs fully on your computer using keyword matching. No embeddings, no vector database, no GPU required. You only need one free Google API key for the final answer.

- **Stack:** Python, rank-bm25, pypdf, Google Gemini Flash, Gradio
- **Key concept:** BM25 keyword search as a replacement for vector embeddings
- **What makes it interesting:** You learn that RAG does not always need expensive embedding APIs. A fast keyword index is enough for many use cases.
- **API keys needed:** Google Gemini (free tier)
- **Difficulty:** Beginner

---

#### `rag/semantic-vector-bot`

Upload a PDF and ask questions about it using semantic search. The app understands the meaning of your question, not just the exact words. The embedding model runs locally on your CPU so only one API key is needed.

- **Stack:** Python, ChromaDB, all-MiniLM-L6-v2 (ONNX), Google Gemini Flash, Gradio
- **Key concept:** Vector embeddings and cosine similarity search
- **What makes it interesting:** You learn how text gets turned into numbers, how those numbers are compared, and how a persistent vector database works.
- **API keys needed:** Google Gemini (free tier)
- **Difficulty:** Intermediate

---

#### `rag/enterprise-search`

Multi-tenant knowledge base search where different users see different documents based on their access level.

- **Stack:** Python, Qdrant, FastAPI, LangChain, OAuth2
- **Key concepts:** Multi-source ingestion, per-user access-aware retrieval, Cohere re-ranking
- **Difficulty:** Advanced

---

### MCP (Model Context Protocol)

MCP is a standard way to connect AI models to external tools and data sources. Think of it as a plugin system for AI assistants.

---

#### `mcp/github-server`

An MCP server that gives any MCP-compatible AI assistant access to GitHub operations like listing repos, reading files, and creating issues.

- **Stack:** Python, MCP SDK, GitHub API, asyncio
- **Key concepts:** MCP tool and resource primitives, stdio and SSE transport, OAuth handling
- **Difficulty:** Intermediate

---

#### `mcp/postgres-server`

An MCP server that lets an AI assistant query a Postgres database safely, with read-only enforcement and schema discovery.

- **Stack:** Python, MCP SDK, PostgreSQL, asyncpg
- **Key concepts:** Read-only enforcement, schema discovery as a resource, query validation
- **Difficulty:** Intermediate

---

### Agentic AI

Agents are AI systems that can plan, use tools, and take multiple steps to complete a task. They loop until they have a good enough answer.

---

#### `agentic/research-agent`

An autonomous agent that takes a research question, searches the web, reads PDFs, and produces a structured report with sources.

- **Stack:** Python, OpenAI, Tavily, LangGraph
- **Key concepts:** ReAct loop, web search and PDF tools, source deduplication, structured output
- **Difficulty:** Advanced

---

#### `agentic/code-review-agent`

A multi-agent system that reviews GitHub pull requests. Multiple specialist agents check different things in parallel, then a coordinator writes the final review.

- **Stack:** Python, OpenAI, GitHub API, LangGraph
- **Key concepts:** Orchestrator-worker pattern, parallel agents, human-approval gate
- **Difficulty:** Advanced

---

## Getting started

Every project has its own README with full setup steps. The general pattern is the same for all of them:

```bash
cd rag/bm25-keyword-bot        # or whichever project you want to run

python -m venv venv

# Mac or Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env      # open .env and add your API keys

python main.py
```

---

## What API keys do you need?

| Project | Keys needed | Cost |
|---|---|---|
| rag/bm25-keyword-bot | Google Gemini | Free |
| rag/semantic-vector-bot | Google Gemini | Free |
| rag/enterprise-search | Cohere | Free tier available |
| mcp/github-server | GitHub token | Free |
| mcp/postgres-server | None (local DB) | Free |
| agentic/research-agent | OpenAI, Tavily | Paid |
| agentic/code-review-agent | OpenAI, GitHub | Paid |

Get a free Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). No credit card required.

---

## Prerequisites

- Python 3.11 or newer
- Git
- Docker (optional, only needed for containerised projects)

---

## License

MIT
