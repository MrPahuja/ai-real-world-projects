# Smart File Cabinet: Multi-LLM Personal Search & Study Assistant with OCR & Access Control

This project implements a production-grade **Hybrid RAG (Retrieval-Augmented Generation)** pipeline featuring **User-level Access Control Lists (ACLs)** and **Multimodal OCR fallback** for scanned documents. 

It is designed to give you a transparent look at what happens behind the scenes of search engines, showing you each algorithmic step (indexing, embedding, retrieval, Reciprocal Rank Fusion, and user filtering) in pure Python using a highly relatable **personal study vault** theme.

---

## System Architecture

The search and retrieval flow runs through the following sequence:

```
                  User Query + User Credentials
                               |
             +-----------------+-----------------+
             |                                   |
     Lexical / Keyword Search           Semantic Vector Search
     (BM25 on Cabinet Chunks)        (Qdrant + gemini-embedding-001)
             |                                   |
             +-----------------+-----------------+
                               |
                   Reciprocal Rank Fusion (RRF)
                               |
                   Access Control List (ACL) Filter
                 (Allows access check per document)
                               |
                     Top-5 Chunks -> LLM
                 (Gemini, OpenAI, or OpenRouter)
                               |
                   Streamed Answer + Citations
```

---

## Key Search & Retrieval Concepts Implemented

### 1. Ingestion & Multimodal OCR
* **The Problem:** Many scanned PDF documents (like handwritten notes, study guides, or scanned receipts) contain no selectable digital text. Standard libraries like `pypdf` fail completely, extracting empty strings.
* **Our Solution:** The ingestion pipeline in `main.py` first checks the length of extracted text. If it is empty or extremely short (under 50 characters), it automatically uploads the document to the **Google Gemini Files API** and runs a multimodal transcription model. This extracts all text, formats tables into Markdown, and cleans up after ingestion.
* **Production Comparison:** 
  * In large cloud architectures, teams use **Azure AI Document Intelligence**, **AWS Textract**, or **GCP Document AI** to perform structure-aware layout parsing.
  * For private or on-premises networks, developers host deep-learning OCR models (like **Surya** or **EasyOCR**) on GPU clusters.

### 2. Hybrid Retrieval (Sparse + Dense)
* **Keyword Search (BM25):** Ideal for exact-term queries like names, specific numbers, error codes, or legal citations (e.g., matching "July 12" or "photosynthesis").
* **Vector Search (Qdrant):** Captures semantic meaning and conceptual queries. If a user asks "what did I buy for my family last month?", vector search retrieves the birthday journal entry even if it does not contain the word "buy".
* **Reciprocal Rank Fusion (RRF):** Instead of normalizing raw, incompatible similarity scores, RRF merges the ranks of matching items using the formula:
$$\text{Score}(\text{chunk}) = \sum_{m \in \{\text{BM25}, \text{Vector}\}} \frac{1}{\text{rank}_m + 60}$$
This guarantees that items appearing high in both retrieval lists are pushed to the top of the context block.

### 3. Access Control Lists (ACLs)
* **Data Security:** Personal vaults require strict security. A visiting guest must not be allowed to see your private journal logs, but they should be able to query public study books.
* **Retrieval-Time Security:** The system checks credentials (`user_id` and `tenant_id`) *at the retrieval level*. Chunks that the user is not explicitly allowed to view are filtered out before being sent to the LLM.

### 4. Neural Rerankers (The Next Step)
* **What is it?** A second-pass filtering mechanism that uses a Cross-Encoder model (e.g., Cohere Rerank or MS-MARCO Cross-Encoder) to read the query and retrieved document chunks jointly, scoring them with high precision.
* **Why it matters:** It eliminates irrelevant noise, prevents "lost in the middle" context window issues, and optimizes answer accuracy.
* **Enterprise equivalent:** Azure AI Search calls this feature the "Semantic Ranker".

---

## LLM Provider Comparison & Cost Choices

This project supports multiple LLM endpoints. Choose the provider that best fits your development budget and quality needs:

| LLM Provider | Supported Models | Cost | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Google Gemini** | `gemini-1.5-flash-latest`, `gemini-1.5-pro-latest` | **100% Free** (within free tier RPM limits) | Free tier requires no credit card; outstanding native multimodal OCR. | Occasional rate limit limits (15 RPM). |
| **OpenRouter** | `meta-llama/llama-3-8b-instruct:free`, `google/gemma-2-9b-it:free`, `zhipu/glm-5.2`, `moonshotai/kimi-k2.6` | **Free & Paid options** | Allows swap-in of open-source and proprietary models; unified keys/billing; prevents vendor lock-in. | Model availability and latency can fluctuate. |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | **Paid** (requires API credit balance) | Industry gold-standard reasoning; high reliability. | No free tier; billing requires active credit card. |

---

## Setup & Configuration

### 1. Installation
Install the necessary python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in the properties:
```env
# Choose provider: "gemini", "openai", or "openrouter"
LLM_PROVIDER=gemini

# Model name:
# e.g. for gemini: "gemini-2.5-flash"
# e.g. for openai: "gpt-4o"
# e.g. for openrouter: "meta-llama/llama-3-8b-instruct:free" or "zhipu/glm-5.2"
CHAT_MODEL=gemini-2.5-flash

# Provide key corresponding to your selected LLM_PROVIDER
# Note: GEMINI_API_KEY is also required for embeddings and OCR fallback.
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 3. Run the App
Launch the interactive web UI:
```bash
python main.py
```
Open the local Gradio link shown in the terminal (usually `http://127.0.0.1:7860`).

---

## Project Structure
* `main.py`: Main application code containing Qdrant/BM25 indexing, retrieval-fusion, user credentials filtering, and the Gradio web UI.
* `requirements.txt`: Python package requirements.
* `.env.example`: Configuration variables template.
