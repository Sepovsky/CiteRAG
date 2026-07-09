#!/usr/bin/env python3
"""Benchmark CiteRAG accuracy and end-to-end latency on any PDF in the eval set.
Supports --kg flag to enable knowledge-graph augmented retrieval with audit trails."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from langchain_ollama import ChatOllama
from ingest import load_document, split_document
from retrieve import HybridRetriever

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent


@dataclass
class AuditTrailResult:
    query_entities: list[str] = field(default_factory=list)
    expanded_entities: list[str] = field(default_factory=list)
    graph_paths: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)


@dataclass
class QuestionResult:
    id: str
    question: str
    answer: str
    correct: bool
    latency_s: float
    source_pages: list
    scoring_notes: str
    audit_trail: AuditTrailResult | None = None


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


def format_context(docs) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_prompt(context: str, query: str) -> str:
    return f"""You are CiteRAG, a citation-grounded document Q&A assistant.

Answer using only the retrieved context below. Every claim must be traceable to the cited pages.

Context:
{context}

Question:
{query}

Instructions:
- Answer only from the context. Do not use outside knowledge.
- If the answer is not in the context, say: "I could not find the answer in the document."
- Be concise (2-4 sentences).
- End with a line: "Citations: Page X, Page Y" listing every page you used.
"""


async def answer_question(
    retriever: HybridRetriever | "HybridGraphRetriever", llm: ChatOllama, question: str, use_kg: bool = False
) -> tuple[str, list, float, AuditTrailResult | None]:
    start = time.perf_counter()
    if use_kg:
        docs, audit = await asyncio.to_thread(retriever.retrieve, question)
        audit_result = AuditTrailResult(
            query_entities=audit.query_entities,
            expanded_entities=audit.expanded_entities,
            graph_paths=audit.triples_used,
            retrieved_chunks=[{"page": p} for p in audit.retrieved_pages],
        )
    else:
        docs = await asyncio.to_thread(retriever.retrieve, question)
        audit_result = None
    context = format_context(docs)
    response = await llm.ainvoke(build_prompt(context, question))
    elapsed = time.perf_counter() - start
    pages = sorted({doc.metadata.get("page", "N/A") for doc in docs})
    return response.content, pages, elapsed, audit_result


async def main_async(pdf_path: Path, use_kg: bool = False) -> int:
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
    pages = load_document(str(pdf_path))
    splits = split_document(pages)
    print(f"PDF: {pdf_path.name} — {len(pages)} pages -> {len(splits)} chunks (800 char, 150 overlap)")

    # --- Build retriever once, share across all questions ---
    build_start = time.perf_counter()
    retriever = HybridRetriever(str(pdf_path), force_rebuild=False)
    build_s = time.perf_counter() - build_start
    print(f"Index build / cache load: {build_s:.2f}s")

    # --- Optionally build / load knowledge graph ---
    kg_build_s = 0.0
    kg = None
    if use_kg:
        from graph_rag import HybridGraphRetriever, build_kg_for_document
        kg_start = time.perf_counter()
        kg = build_kg_for_document(str(pdf_path))
        kg_build_s = time.perf_counter() - kg_start
        print(f"KG build / load: {kg_build_s:.2f}s ({kg.stats})")
        retriever = HybridGraphRetriever(str(pdf_path), kg=kg)

    llm = ChatOllama(model="llama3.1", temperature=0.0)

    # --- Run questions ---
    results: list[QuestionResult] = []
    latencies: list[float] = []

    for spec in eval_questions:
        print(f"\nQ: {spec['question']}")
        answer, pages_used, latency, audit = await answer_question(
            retriever, llm, spec["question"], use_kg=use_kg
        )
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
                audit_trail=audit,
            )
        )
        mark = "PASS" if correct else "FAIL"
        audit_info = ""
        if audit and audit.query_entities:
            audit_info = f" | KG entities: {audit.query_entities}"
        print(f"  [{mark}] {latency:.2f}s | pages={pages_used}{audit_info}")
        print(f"  {answer.strip()[:220]}...")

    correct_count = sum(1 for r in results if r.correct)
    total = len(results)

    # --- Cold-start sample ---
    print("\nMeasuring cold-start latency (rebuilds index)...")
    from qa import ask_document
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
        "use_kg": use_kg,
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
    if use_kg:
        summary["kg_build_or_load_s"] = round(kg_build_s, 2)
        try:
            summary["kg_stats"] = kg.kg.stats if use_kg else ""
        except Exception:
            pass

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
    parser = argparse.ArgumentParser(description="Benchmark CiteRAG accuracy")
    parser.add_argument("pdf_path", nargs="?", default=None, help="Path to document")
    parser.add_argument("--kg", action="store_true", help="Enable knowledge graph augmentation")
    args = parser.parse_args()

    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
    else:
        pdf_path = ROOT / "data" / "sigcomm16_cs2p.pdf"

    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return 1

    return asyncio.run(main_async(pdf_path, use_kg=args.kg))


if __name__ == "__main__":
    raise SystemExit(main())
