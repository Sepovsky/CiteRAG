"""Knowledge-graph-augmented retrieval with entity extraction, graph traversal, and audit trails."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import networkx as nx
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from ingest import load_document, split_document
from retrieve import HybridRetriever

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """You are a knowledge graph extractor. Extract entities and their relations from the given document chunk.
Output ONLY a JSON array of objects with these fields:
- subject: the entity name (short, canonical)
- subject_type: type of the entity (e.g., Protocol, Algorithm, Dataset, Concept, Metric, Organization, Person)
- predicate: the relationship in UPPER_CASE (e.g., USES, IMPLEMENTS, ACHIEVES, PROPOSES, EVALUATES)
- object: the related entity name
- object_type: type of the related entity

Rules:
- Extract domain-specific technical entities (protocols, algorithms, datasets, metrics, models, systems).
- Use short canonical names (e.g., "CS2P" not "CS2P system"; "HMM" not "Hidden Markov Model").
- Only extract explicit, factual relations present in the text.
- If no relations are found, output an empty array [].
- Output ONLY the JSON array, no other text."""


@dataclass(frozen=True)
class KGEntity:
    id: str
    label: str
    type: str
    source_page: int
    source_doc: str


@dataclass(frozen=True)
class KGRelation:
    subject_id: str
    predicate: str
    object_id: str
    source_page: int
    source_doc: str


@dataclass
class AuditTrail:
    query_entities: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    expanded_entities: list[str] = field(default_factory=list)
    triples_used: list[dict] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)


def _normalize_entity_id(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = re.sub(r"\s+", "_", name.strip().lower())
    return name


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._entity_to_pages: dict[str, set[int]] = defaultdict(set)

    def add_entity(self, entity: KGEntity) -> None:
        self.graph.add_node(entity.id, label=entity.label, type=entity.type)
        self._entity_to_pages[entity.id].add(entity.source_page)

    def add_relation(self, relation: KGRelation) -> None:
        if not self.graph.has_edge(relation.subject_id, relation.object_id):
            self.graph.add_edge(
                relation.subject_id,
                relation.object_id,
                predicate=relation.predicate,
                source=relation.source_doc,
            )

    def find_entities(self, query: str) -> list[str]:
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[str, float]] = []
        for node_id in self.graph.nodes:
            label = self.graph.nodes[node_id].get("label", node_id).lower()
            label_words = set(label.split())

            overlap = len(query_words & label_words)
            if overlap > 0:
                scored.append((node_id, overlap))

            if query_lower in label or label in query_lower:
                scored.append((node_id, 10.0))

            if node_id in query_lower or query_lower in node_id:
                scored.append((node_id, 5.0))

        seen: set[str] = set()
        result: list[str] = []
        for node_id, _ in sorted(scored, key=lambda x: -x[1]):
            if node_id not in seen:
                seen.add(node_id)
                result.append(node_id)
        return result

    def expand(self, entity_ids: list[str], max_depth: int = 1) -> list[str]:
        expanded: list[str] = list(entity_ids)
        for eid in entity_ids:
            if eid not in self.graph:
                continue
            predecessors = list(self.graph.predecessors(eid))
            successors = list(self.graph.successors(eid))
            for neighbor in predecessors + successors:
                if neighbor not in expanded:
                    expanded.append(neighbor)
        return expanded

    def get_triples(self, entity_ids: list[str]) -> list[dict]:
        triples: list[dict] = []
        for u, v, data in self.graph.edges(data=True):
            if u in entity_ids or v in entity_ids:
                triples.append({
                    "subject": u,
                    "predicate": data.get("predicate", ""),
                    "object": v,
                })
        return triples

    def get_pages_for_entities(self, entity_ids: list[str]) -> list[int]:
        pages: set[int] = set()
        for eid in entity_ids:
            pages.update(self._entity_to_pages.get(eid, set()))
        return sorted(pages)

    @property
    def stats(self) -> str:
        return f"{self.graph.number_of_nodes()} entities, {self.graph.number_of_edges()} relations"

    def save(self, path: str) -> None:
        data = {
            "nodes": [],
            "edges": [],
            "entity_to_pages": {k: list(v) for k, v in self._entity_to_pages.items()},
        }
        for n, attrs in self.graph.nodes(data=True):
            data["nodes"].append({"id": n, **attrs})
        for u, v, attrs in self.graph.edges(data=True):
            data["edges"].append({"subject": u, "object": v, **attrs})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"KG saved to {path}")

    @classmethod
    def load(cls, path: str) -> Optional[KnowledgeGraph]:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        kg = cls()
        for n in data.get("nodes", []):
            kg.graph.add_node(n["id"], label=n.get("label", n["id"]), type=n.get("type", ""))
        for e in data.get("edges", []):
            kg.graph.add_edge(
                e["subject"], e["object"],
                predicate=e.get("predicate", ""),
                source=e.get("source", ""),
            )
        for eid, pages in data.get("entity_to_pages", {}).items():
            kg._entity_to_pages[eid] = set(pages)
        logger.info(f"KG loaded from {path}: {kg.stats}")
        return kg


class GraphExtractor:
    def __init__(self, model_name: str = "llama3.1", temperature: float = 0.0):
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def extract(self, splits: list[Document], source_doc: str, batch_size: int = 5) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            kg = self._extract_batch(kg, batch, source_doc)
        return kg

    def _extract_batch(self, kg: KnowledgeGraph, batch: list[Document], source_doc: str) -> KnowledgeGraph:
        context_parts = []
        for doc in batch:
            page = doc.metadata.get("page", 1)
            context_parts.append(f"[Page {page}]\n{doc.page_content}")

        prompt = f"""{EXTRACT_SYSTEM_PROMPT}

Document chunks:
{'---'.join(context_parts)}
"""
        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            triples = json.loads(raw)
        except Exception as e:
            logger.warning(f"Batch extraction failed: {e}")
            return kg

        if not isinstance(triples, list):
            return kg

        for t in triples:
            try:
                subj_name = str(t.get("subject", "")).strip()
                obj_name = str(t.get("object", "")).strip()
                predicate = str(t.get("predicate", "")).strip().upper()
                if not subj_name or not obj_name or not predicate:
                    continue

                subj_id = _normalize_entity_id(subj_name)
                obj_id = _normalize_entity_id(obj_name)
                subj_type = str(t.get("subject_type", "Concept"))
                obj_type = str(t.get("object_type", "Concept"))
                page = 1

                for doc in batch:
                    dp = doc.metadata.get("page", 1)
                    if subj_name.lower() in doc.page_content.lower():
                        page = dp
                        break

                subj_entity = KGEntity(id=subj_id, label=subj_name, type=subj_type, source_page=page, source_doc=source_doc)
                obj_entity = KGEntity(id=obj_id, label=obj_name, type=obj_type, source_page=page, source_doc=source_doc)
                relation = KGRelation(subject_id=subj_id, predicate=predicate, object_id=obj_id, source_page=page, source_doc=source_doc)

                kg.add_entity(subj_entity)
                kg.add_entity(obj_entity)
                kg.add_relation(relation)
            except Exception as e:
                logger.warning(f"Failed to add triple {t}: {e}")

        return kg


class GraphRetriever:
    def __init__(self, kg: KnowledgeGraph, splits: list[Document]):
        self.kg = kg
        self.splits = splits

    def retrieve(self, query: str) -> tuple[list[Document], AuditTrail]:
        audit = AuditTrail()
        entities = self.kg.find_entities(query)
        audit.query_entities = entities
        if not entities:
            return [], audit

        expanded = self.kg.expand(entities[:5], max_depth=1)
        audit.expanded_entities = expanded

        triples = self.kg.get_triples(expanded)
        audit.triples_used = triples

        target_pages = self.kg.get_pages_for_entities(expanded)
        audit.retrieved_pages = target_pages

        matched = [doc for doc in self.splits if doc.metadata.get("page") in target_pages]
        return matched, audit


class HybridGraphRetriever:
    def __init__(self, pdf_path: str, kg: KnowledgeGraph, force_rebuild: bool = False):
        self.hybrid_retriever = HybridRetriever(pdf_path, force_rebuild=force_rebuild)
        _, ext = Path(pdf_path).suffix.lower(), None
        docs = load_document(pdf_path)
        self.splits = split_document(docs)
        self.graph_retriever = GraphRetriever(kg, self.splits)

    def retrieve(self, query: str) -> tuple[list[Document], AuditTrail]:
        hybrid_docs = self.hybrid_retriever.retrieve(query)
        graph_docs, audit = self.graph_retriever.retrieve(query)

        seen: set[str] = set()
        merged: list[Document] = []
        for doc in hybrid_docs + graph_docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged, audit


def build_kg_for_document(pdf_path: str, force_rebuild: bool = False) -> KnowledgeGraph:
    stem = Path(pdf_path).stem
    kg_path = Path("knowledge_graph") / f"{stem}_kg.json"
    if not force_rebuild:
        cached = KnowledgeGraph.load(str(kg_path))
        if cached is not None:
            return cached

    docs = load_document(pdf_path)
    splits = split_document(docs)
    logger.info(f"Extracting KG from {len(splits)} chunks...")
    extractor = GraphExtractor()
    kg = extractor.extract(splits, source_doc=pdf_path, batch_size=5)
    kg.save(str(kg_path))
    return kg
