# CiteRAG — Citation-Grounded RAG for Document QA

> Verifiable, source-grounded answers from PDF documents — fully local, no API key required.

**CiteRAG** is an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline with **hybrid retrieval (dense + sparse)**, cross-encoder reranking, and **page-level citations**. Built with LangChain, FAISS, BM25, Hugging Face Transformers, and Ollama for fully local, grounded inference.


---

## How it works

```
PDF → Load (pdfplumber) → Chunk → Embed (MiniLM-L6)
                                        ↓
               User question → Dense retrieval (FAISS, top-10)
                                       +
                              Sparse retrieval (BM25, top-10)
                                        ↓
                           Reciprocal Rank Fusion
                                        ↓
                           Cross-encoder reranker → Top-3
                                        ↓
               Ollama (Llama 3.1) → Citation-grounded answer + page sources
```

---

## Features

- End-to-end RAG pipeline with hybrid retrieval (dense + sparse)
- Cross-encoder reranking for high-precision top-3 results
- Page-level citations on every answer for verifiable, source-grounded responses
- pdfplumber for robust PDF text and table extraction
- Custom Transformer embeddings (avoids broken `sentence-transformers` dependency)
- Persistent FAISS vector store with PDF fingerprint caching
- Fully offline inference via Ollama — no cloud API key needed

---

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| PDF parsing | pdfplumber |
| Dense embeddings | MiniLM-L6 (HuggingFace Transformers) |
| Sparse retrieval | BM25 via `rank-bm25` |
| Fusion | Reciprocal Rank Fusion |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector store | FAISS |
| LLM runtime | Ollama |
| Model | Llama 3.1 |

---

## Project structure

```
CiteRAG/
├── app/
│   ├── ingest.py       # Load PDF via pdfplumber and split into chunks
│   ├── retrieve.py     # Hybrid retriever (FAISS + BM25 + reranker)
│   ├── qa.py           # Async citation-grounded Q&A with streaming
│   └── main.py         # Async interactive CLI with streaming output
├── data/
│   └── sample.pdf
├── scripts/
│   └── benchmark.py    # Accuracy and latency eval
├── vector_store/
│   └── faiss_*         # Cached FAISS indexes (per PDF fingerprint)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sepovsky/CiteRAG.git
cd CiteRAG
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
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
```

> For macOS or Windows, download from [ollama.com](https://ollama.com).

### 5. Add your PDF

```bash
cp your-document.pdf data/
```

---

## Usage

Run each step individually to test, or jump straight to the interactive CLI:

```bash
# Step 1 — test PDF loading and chunking
python app/ingest.py data/your-document.pdf

# Step 2 — test hybrid retrieval
python app/retrieve.py data/your-document.pdf

# Step 3 — test citation-grounded Q&A
python app/qa.py

# Step 4 — interactive CLI (streaming)
python app/main.py
```

**Example questions to try:**

- What problem does this paper solve?
- What method is proposed?
- What are the main contributions?
- What dataset or traces are used?
- What are the key results?

---

## Benchmark

Run the accuracy and latency benchmark (requires Ollama and a PDF in `data/`):

```bash
python scripts/benchmark.py
```

---

## License

For educational and personal use.
