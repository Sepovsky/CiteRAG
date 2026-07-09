"""Persistent hybrid retrieval: FAISS (dense) + BM25 (sparse) with cross-encoder reranking.

Uses HuggingFace transformers directly to avoid broken sentence_transformers deps.
"""

import hashlib
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from ingest import load_document, split_document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_STORE_DIR = "vector_store"
TOP_K_DENSE = 10
TOP_K_SPARSE = 10
TOP_K_RERANK = 3


# ---------------------------------------------------------------------------
# Transformer-based embeddings (drop-in for HuggingFaceEmbeddings)
# ---------------------------------------------------------------------------

class TransformersEmbeddings(Embeddings):
    """LangChain-compatible embeddings using transformers (avoiding sentence_transformers)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _embed(self, text: str) -> List[float]:
        encoded = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        # Mean pooling
        mask = encoded["attention_mask"][..., None].float()
        embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding[0].cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Cross-encoder for reranking using transformers directly."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, pairs: List[tuple]) -> List[float]:
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        scores = outputs.logits.squeeze(-1).cpu().numpy().tolist()
        return scores if isinstance(scores, list) else [scores]


# ---------------------------------------------------------------------------
# Persistent vector-store helpers
# ---------------------------------------------------------------------------

def _pdf_fingerprint(pdf_path: str) -> str:
    path = Path(pdf_path)
    mtime = str(path.stat().st_mtime)
    return hashlib.md5(f"{path.absolute()}:{mtime}".encode()).hexdigest()[:12]


def _faiss_path(pdf_path: str) -> Path:
    return Path(VECTOR_STORE_DIR) / f"faiss_{_pdf_fingerprint(pdf_path)}"


def _load_or_build_splits(pdf_path: str) -> List[Document]:
    docs = load_document(pdf_path)
    return split_document(docs)


def build_vectorstore(pdf_path: str, force_rebuild: bool = False) -> FAISS:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    fp = _faiss_path(pdf_path)

    if not force_rebuild and fp.exists():
        embeddings = TransformersEmbeddings()
        return FAISS.load_local(str(fp), embeddings, allow_dangerous_deserialization=True)

    splits = _load_or_build_splits(pdf_path)
    embeddings = TransformersEmbeddings()
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(str(fp))
    return vs


def _build_bm25(splits: List[Document]) -> BM25Okapi:
    tokenized = [doc.page_content.split() for doc in splits]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(result_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion is a technique for combining multiple retrieval results into a single ranked list.
    It is based on the idea that the more relevant a document is to a query, the higher its rank should be.
    It is a simple and effective way to combine multiple retrieval results into a single ranked list.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for docs in result_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


# ---------------------------------------------------------------------------
# Hybrid retriever with reranking
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Dense + sparse hybrid retriever with cross-encoder reranking."""

    def __init__(self, pdf_path: str, force_rebuild: bool = False):
        self.pdf_path = pdf_path
        self.vectorstore = build_vectorstore(pdf_path, force_rebuild=force_rebuild)
        self.splits = _load_or_build_splits(pdf_path)
        self.bm25 = _build_bm25(self.splits)
        self._reranker: CrossEncoderReranker | None = None

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def retrieve(self, query: str) -> List[Document]:
        dense_docs = self.vectorstore.similarity_search(query, k=TOP_K_DENSE)

        tok_q = query.split()
        bm25_scores = self.bm25.get_scores(tok_q)
        top_sp = np.argsort(bm25_scores)[-TOP_K_SPARSE:][::-1]
        sparse_docs = [self.splits[i] for i in top_sp if bm25_scores[i] > 0]

        fused = _rrf([dense_docs, sparse_docs])
        if not fused:
            return fused

        pairs = [(query, d.page_content) for d in fused]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:TOP_K_RERANK]]

    def invoke(self, query: str) -> List[Document]:
        return self.retrieve(query)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sigcomm16_cs2p.pdf"
    retriever = HybridRetriever(path)
    query = "What dataset or trace is used?"
    results = retriever.retrieve(query)
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "N/A")
        print(f"\n--- Result {i} (Page {page}) ---")
        print(doc.page_content[:600])
