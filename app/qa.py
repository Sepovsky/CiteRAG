from langchain_ollama import ChatOllama
from retrieve import get_retriever

def format_context(docs):
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        parts.append(f"Page {page}:\n{doc.page_content}")
    return "\n\n".join(parts)

def ask_pdf(pdf_path: str, query: str):
    retriever = get_retriever(pdf_path)
    retrieved_docs = retriever.invoke(query)
    context = format_context(retrieved_docs)

    llm = ChatOllama(
        model="llama3.1",
        temperature=0.0,
    )

    prompt = f"""
You are answering questions using only the provided PDF context.

Context:
{context}

Question:
{query}

Instructions:
- Answer only from the context.
- If the answer is not in the context, say: "I could not find the answer in the PDF."
- At the end, list the page numbers you used.
"""

    response = llm.invoke(prompt)
    return response.content, retrieved_docs


if __name__ == "__main__":
    pdf_path = "data/sigcomm16_cs2p.pdf"
    question = "What problem does this paper solve?"

    answer, docs = ask_pdf(pdf_path, question)

    print("\nAnswer:\n")
    print(answer)

    print("\nRetrieved Chunks:\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:700])
        print()