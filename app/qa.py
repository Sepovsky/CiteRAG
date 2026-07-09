"""Citation-grounded question answering with async streaming support."""

import asyncio
from typing import AsyncGenerator, List, Tuple

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from retrieve import HybridRetriever


def format_context(docs: List[Document]) -> str:
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
- Be concise (2–4 sentences).
- End with a line: "Citations: Page X, Page Y" listing every page you used.
"""


async def ask_document(
    document_path: str, query: str, force_rebuild: bool = False
) -> Tuple[str, List[Document]]:
    """Retrieve relevant chunks and return a citation-grounded answer."""
    retriever = await asyncio.to_thread(HybridRetriever, document_path, force_rebuild)
    retrieved_docs = await asyncio.to_thread(retriever.retrieve, query)
    context = format_context(retrieved_docs)

    llm = ChatOllama(model="llama3.1", temperature=0.0)
    response = await llm.ainvoke(build_prompt(context, query))
    return response.content, retrieved_docs


async def stream_answer(
    document_path: str, query: str, force_rebuild: bool = False
) -> AsyncGenerator[str | Tuple[str, List[Document]], None]:
    """Stream answer tokens, then yield ('__docs__', retrieved_docs) as final item."""
    retriever = await asyncio.to_thread(HybridRetriever, document_path, force_rebuild)
    retrieved_docs = await asyncio.to_thread(retriever.retrieve, query)
    context = format_context(retrieved_docs)

    llm = ChatOllama(model="llama3.1", temperature=0.0)
    async for chunk in llm.astream(build_prompt(context, query)):
        yield chunk.content
    yield ("__docs__", retrieved_docs)


ask_pdf = ask_document


if __name__ == "__main__":
    pdf_path = "data/sigcomm16_cs2p.pdf"
    question = "What problem does this paper solve?"

    async def _demo():
        answer, docs = await ask_document(pdf_path, question)
        print("\nAnswer:\n")
        print(answer)
        print("\nRetrieved chunks:\n")
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "N/A")
            print(f"--- Chunk {i} (Page {page}) ---")
            print(doc.page_content[:700])
            print()

    asyncio.run(_demo())
