"""Semantic retrieval over document chunks with FAISS and Hugging Face embeddings."""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_pdf, split_pdf

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def build_vectorstore(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_pdf(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore


def get_retriever(pdf_path: str):
    vectorstore = build_vectorstore(pdf_path)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


if __name__ == "__main__":
    retriever = get_retriever("data/sigcomm16_cs2p.pdf")

    query = "What dataset or trace is used?"
    results = retriever.invoke(query)

    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "N/A")
        print(f"\n--- Result {i} (Page {page}) ---")
        print(doc.page_content[:700])
