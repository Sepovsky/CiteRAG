# RAG PDF Q&A with LangChain

> Ask questions about any PDF — fully local, no API key required.

A clean, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, FAISS, local embeddings, and Ollama. Designed for learning the core mechanics of RAG in a practical, readable codebase.

---

## How it works

```
PDF → Load → Chunk → Embed → FAISS index
                                   ↓
              User question → Retrieve top-k chunks
                                   ↓
                       Ollama (Llama 3.1) → Answer + source pages
```

---

## Features

- Parse and chunk PDF documents
- Generate embeddings locally via Hugging Face `sentence-transformers`
- Store and search vectors with FAISS
- Answer questions using a local LLM through Ollama
- Display source page numbers for every answer
- Runs entirely offline — no API key needed

---

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| PDF parsing | PyPDF |
| Embeddings | sentence-transformers (HuggingFace) |
| Vector store | FAISS |
| LLM runtime | Ollama |
| Model | Llama 3.1 |

---

## Project structure

```
rag_pdf_langchain/
├── app/
│   ├── ingest.py       # Load PDF and split into chunks
│   ├── retrieve.py     # Build FAISS index and retrieve chunks
│   ├── qa.py           # Combine retrieval + LLM to answer questions
│   └── main.py         # Interactive CLI loop
├── data/
│   └── sample.pdf
├── vector_store/
├── .env
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create project folders

```bash
git clone <your-repo-url>
cd rag_pdf_langchain
mkdir -p data vector_store
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model

```bash
# Install (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version

# Pull Llama 3.1
ollama pull llama3.1
```

> For macOS or Windows, download from [ollama.com](https://ollama.com).

### 5. Add your PDF

Place any PDF in the `data/` folder:

```bash
cp your-document.pdf data/
```

---

## Usage

Run each step individually to test, or jump straight to the interactive CLI:

```bash
# Step 1 — test PDF loading and chunking
python app/ingest.py

# Step 2 — test retrieval
python app/retrieve.py

# Step 3 — test question answering
python app/qa.py

# Step 4 — interactive Q&A
python app/main.py
```

**Example questions to try:**

- What problem does this paper solve?
- What method is proposed?
- What are the main contributions?
- What dataset or traces are used?
- What are the key results?
- What limitations are discussed?

---

## Why chunking matters

Sending a full PDF to an LLM is expensive and noisy. Chunking solves this by:

- Breaking the document into smaller, searchable pieces
- Improving retrieval precision
- Reducing irrelevant context sent to the model
- Lowering compute cost and latency

Answer quality is directly tied to retrieval quality — better chunks lead to better answers.

---

## Improving results

Some next steps to explore:

- [ ] Save and reload the FAISS index to avoid rebuilding on every run
- [ ] Try stronger embedding models
- [ ] Retrieve more candidates and apply reranking
- [ ] Tune chunk size and overlap
- [ ] Add hybrid search (keyword + semantic)
- [ ] Support multiple PDFs
- [ ] Add a Streamlit UI
- [ ] Improve page-level citations
- [ ] Test stronger Ollama models (e.g. Llama 3.3, Mistral)

---

## Learning goals

This project is a practical introduction to:

- What RAG is and why it works
- How text chunking affects retrieval quality
- How embeddings represent meaning as vectors
- How vector similarity search works (FAISS)
- How retrieved context grounds LLM answers
- How to build a fully local, privacy-friendly Q&A system

---

## License

For educational and personal use.