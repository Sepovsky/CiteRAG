# CiteRAG — Citation-Grounded RAG for Document QA

> Verifiable, source-grounded answers from PDF documents — fully local, no API key required.

**CiteRAG** is an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline with **semantic retrieval**, document chunking, and **page-level citations**. Built with LangChain, FAISS, Hugging Face embeddings, and Ollama for fully local, grounded inference.

Repository: [github.com/Sepovsky/CiteRAG](https://github.com/Sepovsky/CiteRAG)

---

## How it works

```
PDF → Load → Chunk → Embed → FAISS index
                                   ↓
              User question → Semantic retrieval (top-k)
                                   ↓
              Ollama (Llama 3.1) → Citation-grounded answer + page sources
```

---

## Features

- End-to-end RAG pipeline with semantic chunk retrieval
- Page-level citations on every answer for verifiable, source-grounded responses
- Local Hugging Face `sentence-transformers` embeddings
- FAISS vector search for fast similarity retrieval
- Fully offline inference via Ollama — no cloud API key needed

---

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| PDF parsing | PyPDF |
| Embeddings | sentence-transformers (Hugging Face) |
| Vector store | FAISS |
| LLM runtime | Ollama |
| Model | Llama 3.1 |

---

## Project structure

```
CiteRAG/
├── app/
│   ├── ingest.py       # Load PDF and split into chunks
│   ├── retrieve.py     # FAISS index + semantic retrieval
│   ├── qa.py           # Citation-grounded Q&A
│   └── main.py         # Interactive CLI
├── data/
│   └── sample.pdf
├── scripts/
│   └── benchmark.py    # Accuracy and latency eval
├── vector_store/
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
python app/ingest.py

# Step 2 — test semantic retrieval
python app/retrieve.py

# Step 3 — test citation-grounded Q&A
python app/qa.py

# Step 4 — interactive CLI
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
