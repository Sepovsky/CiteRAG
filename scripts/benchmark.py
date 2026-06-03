#!/usr/bin/env python3
"""Benchmark CiteRAG accuracy and end-to-end latency on the CS2P SIGCOMM paper."""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from langchain_ollama import ChatOllama
from ingest import load_pdf, split_pdf
from retrieve import build_vectorstore
from qa import build_prompt, format_context

PDF_PATH = ROOT / "data" / "sigcomm16_cs2p.pdf"
RESULTS_PATH = Path(__file__).resolve().parent / "benchmark_results.json"

# Ground truth derived from sigcomm16_cs2p.pdf (CS2P, SIGCOMM 2016)
EVAL_QUESTIONS = [
    {
        "id": "problem",
        "question": "What problem does this paper solve?",
        "must_match_any": [
            ["bitrate", "throughput"],
            ["video", "streaming", "internet video"],
            ["adaptation", "selection", "prediction"],
        ],
        "must_not_contain": ["could not find"],
    },
    {
        "id": "cs2p_definition",
        "question": "What is CS2P?",
        "must_match_any": [
            ["cs2p"],
            ["throughput", "prediction"],
            ["data-driven", "data driven"],
        ],
        "must_not_contain": ["could not find"],
    },
    {
        "id": "dataset",
        "question": "What dataset or trace is used?",
        "must_match_any": [
            ["20m", "20 m", "20 million", "20m+"],
            ["session"],
            ["iqiyi", "i qiyi"],
        ],
        "must_not_contain": ["could not find"],
    },
    {
        "id": "midstream_model",
        "question": "What model is used for midstream throughput prediction?",
        "must_match_any": [
            ["hidden markov", "hmm", "markov model"],
        ],
        "must_not_contain": ["could not find"],
    },
    {
        "id": "initial_error_improvement",
        "question": "By how much does CS2P reduce median initial throughput prediction error compared to prior approaches?",
        "must_match_any": [
            ["40%", "40 percent", "forty percent"],
        ],
        "must_not_contain": ["could not find"],
    },
    {
        "id": "midstream_error_improvement",
        "question": "By how much does CS2P reduce median midstream throughput prediction error?",
        "must_match_any": [
            ["50%", "50 percent", "fifty percent"],
        ],
        "must_not_contain": ["could not find"],
    },
    {
        "id": "qoe_improvement",
        "question": "What QoE improvement does CS2P achieve over buffer-based adaptation?",
        "must_match_any": [
            ["14%", "14 percent", "fourteen percent"],
            ["buffer", "bb"],
        ],
        "must_not_contain": ["could not find"],
    },
    {
        "id": "contributions",
        "question": "What are the main contributions of this paper?",
        "must_match_any": [
            ["throughput", "dataset", "20m", "session"],
            ["cs2p", "predictor", "prediction"],
            ["prototype", "evaluation", "experiment", "simulation"],
        ],
        "must_not_contain": ["could not find"],
    },
]


@dataclass
class QuestionResult:
    id: str
    question: str
    answer: str
    correct: bool
    latency_s: float
    source_pages: list
    scoring_notes: str


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def score_answer(answer: str, spec: dict) -> tuple[bool, str]:
    text = normalize(answer)
    notes: list[str] = []

    for forbidden in spec.get("must_not_contain", []):
        if forbidden.lower() in text:
            return False, f"forbidden phrase: {forbidden!r}"

    groups = spec["must_match_any"]
    matched_groups = 0
    for i, group in enumerate(groups):
        if any(term.lower() in text for term in group):
            matched_groups += 1
        else:
            notes.append(f"missing group {i + 1}: {group}")

    required = max(1, len(groups) - 1) if len(groups) > 2 else len(groups)
    ok = matched_groups >= required
    return ok, "; ".join(notes) if notes else "all required concept groups matched"


def answer_question(retriever, question: str) -> tuple[str, list, float]:
    start = time.perf_counter()
    docs = retriever.invoke(question)
    context = format_context(docs)
    llm = ChatOllama(model="llama3.1", temperature=0.0)
    response = llm.invoke(build_prompt(context, question))
    elapsed = time.perf_counter() - start
    pages = sorted({doc.metadata.get("page", "N/A") for doc in docs})
    return response.content, pages, elapsed


def run_cold_latency(pdf_path: Path, question: str) -> float:
    """Current app behavior: rebuild index on every question."""
    from qa import ask_document

    start = time.perf_counter()
    ask_document(str(pdf_path), question)
    return time.perf_counter() - start


def main() -> int:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        return 1

    pages = load_pdf(str(PDF_PATH))
    splits = split_pdf(pages)
    print(f"PDF: {len(pages)} pages -> {len(splits)} chunks (800 char, 150 overlap)")

    index_start = time.perf_counter()
    vectorstore = build_vectorstore(str(PDF_PATH))
    index_build_s = time.perf_counter() - index_start
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"Index build: {index_build_s:.2f}s")

    results: list[QuestionResult] = []
    latencies: list[float] = []

    for spec in EVAL_QUESTIONS:
        print(f"\nQ: {spec['question']}")
        answer, pages_used, latency = answer_question(retriever, spec["question"])
        correct, notes = score_answer(answer, spec)
        latencies.append(latency)
        results.append(
            QuestionResult(
                id=spec["id"],
                question=spec["question"],
                answer=answer.strip(),
                correct=correct,
                latency_s=round(latency, 2),
                source_pages=pages_used,
                scoring_notes=notes,
            )
        )
        mark = "PASS" if correct else "FAIL"
        print(f"  [{mark}] {latency:.2f}s | pages={pages_used}")
        print(f"  {answer.strip()[:220]}...")

    correct_count = sum(1 for r in results if r.correct)
    total = len(results)

    # Cold-start latency: one sample using current qa.py behavior
    print("\nMeasuring cold-start latency (rebuilds index each query, current qa.py)...")
    cold_latency = run_cold_latency(PDF_PATH, EVAL_QUESTIONS[0]["question"])
    print(f"  cold_start_sample: {cold_latency:.2f}s")

    summary = {
        "pdf_pages": len(pages),
        "chunk_count": len(splits),
        "chunk_size": 800,
        "chunk_overlap": 150,
        "top_k": 3,
        "index_build_s": round(index_build_s, 2),
        "accuracy": {
            "correct": correct_count,
            "total": total,
            "rate": round(correct_count / total, 2),
        },
        "warm_query_latency_s": {
            "median": round(statistics.median(latencies), 2),
            "mean": round(statistics.mean(latencies), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "cold_start_latency_s": round(cold_latency, 2),
        "model": "llama3.1",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }

    output = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    print("\n" + "=" * 72)
    print("CiteRAG — CV METRICS (measured)")
    print("=" * 72)
    print(f"Accuracy: {correct_count}/{total} ({100 * correct_count / total:.0f}%)")
    print(f"Warm Q&A latency (index cached): median {summary['warm_query_latency_s']['median']}s")
    print(f"Cold-start Q&A latency (current qa.py): {summary['cold_start_latency_s']}s")
    print(f"Index build: {summary['index_build_s']}s | {len(pages)} pages -> {len(splits)} chunks")

    print("\n--- Suggested CV bullets ---\n")
    print(
        "Built an end-to-end RAG pipeline with semantic retrieval, document chunking, "
        "and page-level citations for verifiable, source-grounded answers."
    )
    print(
        "Engineered the retrieval stack with LangChain, FAISS, and Hugging Face embeddings, "
        "served via Ollama for fully local, grounded inference."
    )
    print(
        f"\n(Eval: {correct_count}/{total} answer accuracy on a 14-page SIGCOMM paper, "
        f"~{summary['warm_query_latency_s']['median']}s end-to-end latency, zero cloud cost.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
