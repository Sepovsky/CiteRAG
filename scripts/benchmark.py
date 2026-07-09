#!/usr/bin/env python3
"""Benchmark CiteRAG accuracy and end-to-end latency on any PDF in the eval set."""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from qa import ask_document, format_context, build_prompt
from ingest import load_pdf, split_pdf
from retrieve import HybridRetriever

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent


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


async def answer_question(pdf_path: str, question: str) -> tuple[str, list, float]:
    start = time.perf_counter()
    answer, docs = await ask_document(pdf_path, question, force_rebuild=False)
    elapsed = time.perf_counter() - start
    pages = sorted({doc.metadata.get("page", "N/A") for doc in docs})
    return answer, pages, elapsed


async def main_async(pdf_path: Path) -> int:
    pdf_name = pdf_path.name

    with open(EVAL_SET_PATH) as f:
        all_evals = json.load(f)

    if pdf_name not in all_evals:
        print(f"ERROR: No eval questions for {pdf_name} in {EVAL_SET_PATH}")
        available = [k for k in all_evals if k != "_description" and k != "_scoring"]
        print(f"Available PDFs: {available}")
        return 1

    eval_questions = all_evals[pdf_name]

    # --- PDF overview ---
    pages = load_pdf(str(pdf_path))
    splits = split_pdf(pages)
    print(f"PDF: {pdf_path.name} — {len(pages)} pages -> {len(splits)} chunks (800 char, 150 overlap)")

    # --- Warm-up / cache build ---
    build_start = time.perf_counter()
    retriever = HybridRetriever(str(pdf_path), force_rebuild=False)
    build_s = time.perf_counter() - build_start
    print(f"Index build / cache load: {build_s:.2f}s")

    # --- Run questions ---
    results: list[QuestionResult] = []
    latencies: list[float] = []

    for spec in eval_questions:
        print(f"\nQ: {spec['question']}")
        answer, pages_used, latency = await answer_question(str(pdf_path), spec["question"])
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

    # --- Cold-start sample ---
    print("\nMeasuring cold-start latency (rebuilds index)...")
    cold_start = time.perf_counter()
    await ask_document(str(pdf_path), eval_questions[0]["question"], force_rebuild=True)
    cold_latency = time.perf_counter() - cold_start
    print(f"  cold_start_sample: {cold_latency:.2f}s")

    # --- Summary ---
    summary = {
        "pdf": pdf_name,
        "pdf_pages": len(pages),
        "chunk_count": len(splits),
        "chunk_size": 800,
        "chunk_overlap": 150,
        "top_k_dense": 10,
        "top_k_sparse": 10,
        "top_k_rerank": 3,
        "index_build_or_load_s": round(build_s, 2),
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
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }

    results_path = RESULTS_DIR / f"benchmark_{pdf_path.stem}.json"
    output = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {results_path}")

    print("\n" + "=" * 72)
    print("CiteRAG — CV METRICS (measured)")
    print("=" * 72)
    print(f"Accuracy: {correct_count}/{total} ({100 * correct_count / total:.0f}%)")
    print(f"Warm Q&A latency (index cached): median {summary['warm_query_latency_s']['median']}s")
    print(f"Cold-start Q&A latency: {summary['cold_start_latency_s']}s")
    print(f"Index build/load: {summary['index_build_or_load_s']}s | {len(pages)} pages -> {len(splits)} chunks")
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = ROOT / "data" / "sigcomm16_cs2p.pdf"

    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return 1

    return asyncio.run(main_async(pdf_path))


if __name__ == "__main__":
    raise SystemExit(main())
