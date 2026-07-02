# Enterprise RAG Search Cabinet: AI Foundation Hybrid Search System

This project implements a production-grade **Hybrid RAG (Retrieval-Augmented Generation)** search architecture featuring **User-level Access Control Lists (ACLs)**, a **Two-Stage Top-K Retrieval Pipeline**, and **Multimodal OCR fallback** capabilities.

It has been completely modularized to match enterprise coding standards, splitting data storage, ingestion, search retrieval, generation logic, and UI blocks into distinct clean components.

---

## 1. Why Hybrid Search is Better

Standard RAG architectures often rely solely on **Semantic Vector Search**. While powerful, pure vector search fails on specific keyword matches, citation IDs, numbers, and technical terms. To solve this, our system implements **Hybrid Search**, which merges two complementary search paradigms:

1. **Lexical / Keyword Search (BM25):** 
   * **Strength:** Excellent at exact keyword matching, numbers, technical terms, and identifier lookup. For example, if a user queries "Directive 2025-A-01" or a release date like "January 15", BM25 retrieves the exact document section with absolute precision.
   * **Weakness:** Cannot understand semantic synonyms, typos, or context.
2. **Semantic Vector Search (Dense Embeddings):** 
   * **Strength:** Understands concept matches, synonyms, and natural phrasing. For example, if a user asks "how are we funding the computing resources?", it matches text describing "compute budgets allocated under government guidelines" even without overlapping keywords.
   * **Weakness:** Often fails on precise names, numbers, or citation codes.

By combining BM25 keyword matching and Vector semantic matching via **Reciprocal Rank Fusion (RRF)**, the system achieves the highest possible retrieval coverage and accuracy.

---

## 2. The Two-Stage Top-K Pipeline

To execute search efficiently at scale, we use a **Two-Stage Top-K Pipeline**:

```
                  User Query + User Credentials
                                |
             +------------------+------------------+
             |                                     |
     Stage 1: BM25 Retrieval              Stage 1: Vector Search
     Retrieves Top-K1 (20 chunks)        Retrieves Top-K1 (20 chunks)
             |                                     |
             +------------------+------------------+
                                |
                                v
                   Access Control List (ACL) Filter
                Filters out unauthorized user chunks
                                |
                                v
                     Rank Merger (RRF Fusion)
                Merges candidates using rank indices
                                |
                                v
                     Stage 2: Final Top-K2 Selection
                 Selects top K2 (3 chunks) for LLM
```

* **Stage 1 (Parallel Retrieval Pool):** The system queries the keyword index and Qdrant vector index in parallel. Each engine retrieves its respective top **20 candidates** (`BM25_POOL = 20`, `VECTOR_POOL = 20`).
* **Credential Filtering (ACL):** Candidates that the active user is not authorized to see are filtered out immediately.
* **Rank Fusion (RRF):** The remaining candidates are merged and scored using the **Reciprocal Rank Fusion (RRF)** formula:
  $$\text{Score}(c) = \sum_{m \in \{\text{BM25}, \text{Vector}\}} \frac{1}{\text{rank}_m(c) + 60}$$
  This ranks chunks based on their relative position in both retrieval lists.
* **Stage 2 (Final Selection):** The RRF-ranked candidate list is sliced to the final top **3 chunks** (`FINAL_TOP_K = 3`), providing a compact, high-relevance context window for the LLM to minimize token costs and prevent "lost in the middle" reasoning drop-offs.

---

## 3. System Architecture & Modular Structure

The codebase is split into the following production-grade files:

* **[config.py](config.py):** System configurations, global constants, model weights parameters, and client declarations.
* **[database.py](database.py):** Database client setup, structured vector schemas, sample database records, and collection diagnostics.
* **[ingestion.py](ingestion.py):** Character-overlapping token chunkers, OCR transcription fallbacks utilizing Gemini Files API, and document indexing.
* **[retrieval.py](retrieval.py):** Multimodal search engines, credential ACL filters, and RRF rank mergers.
* **[llm.py](llm.py):** Generation instruction formatting, chat prompt contexts, and streaming handlers.
* **[app.py](app.py):** GradBlocks layouts, CSS custom themes, local sample file downloads, and interface triggers.
* **[main.py](main.py):** Application entry point initiating database samples and starting the server.

---

## 4. User Access Boundaries (ACL Guide)

Sample documents are preloaded to illustrate realistic corporate boundary structures:

* **`self` (Cabinet Owner):** Authorized to view **My Personal Journal** (private entries), **Biology Group Project** notes, and public summaries.
* **`admin` (System Administrator / Study Partner):** Authorized to view **Biology Group Project** notes (shared workspace) and public summaries.
* **`guest` (Unauthenticated Visitor):** Limited to public documents only (**Alice in Wonderland Summary**).

---

## 5. Setup & Configuration

### 1. Installation
Install the necessary python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in API keys:
```bash
cp .env.example .env
```

### 3. Run the App
Launch the interactive web UI:
```bash
python main.py
```
Open **`http://127.0.0.1:7860`** to interact with the cabinet chat and inspect retrieval calculations.
