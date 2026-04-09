from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_pdf, split_pdf


def build_vectorstore(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_pdf(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore

def get_retriever(pdf_path: str):
    vectorstore = build_vectorstore(pdf_path)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    retriever = get_retriever("data/sigcomm16_cs2p.pdf")

    # query = "What is the main idea of the paper?"
    query = "What dataset or trace is used?"
    results = retriever.invoke(query)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        # print("Metadata:", doc.metadata)
        print(doc.page_content[:700])