"""Citation-grounded question answering over retrieved document chunks."""

from langchain_ollama import ChatOllama
from retrieve import get_retriever


def format_context(docs):
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


def ask_document(document_path: str, query: str):
    """Retrieve relevant chunks and return a citation-grounded answer."""
    retriever = get_retriever(document_path)
    retrieved_docs = retriever.invoke(query)
    context = format_context(retrieved_docs)

    llm = ChatOllama(
        model="llama3.1",
        temperature=0.0,
    )

    response = llm.invoke(build_prompt(context, query))
    return response.content, retrieved_docs


# Backward-compatible alias
ask_pdf = ask_document


if __name__ == "__main__":
    pdf_path = "data/sigcomm16_cs2p.pdf"
    question = "What problem does this paper solve?"

    answer, docs = ask_document(pdf_path, question)

    print("\nAnswer:\n")
    print(answer)

    print("\nRetrieved chunks:\n")
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "N/A")
        print(f"--- Chunk {i} (Page {page}) ---")
        print(doc.page_content[:700])
        print()
