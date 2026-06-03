"""Document ingestion: load PDFs and split into retrieval-ready chunks."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_pdf(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_pdf("data/sigcomm16_cs2p.pdf")
    splits = split_pdf(docs)

    print(f"Loaded pages: {len(docs)}")
    print(f"Total chunks: {len(splits)}")
    print("\nFirst chunk:\n")
    print(splits[0].page_content[:500])
    print("\nMetadata:")
    print(splits[0].metadata)
