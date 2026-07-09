"""Document ingestion: load PDFs via pdfplumber for superior text and table extraction."""

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _table_to_markdown(table: list[list[str | None]]) -> str:
    if not table or not table[0]:
        return ""
    rows = []
    for row in table:
        cells = [str(cell).strip() if cell else "" for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    header = rows[0]
    sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
    rows.insert(1, sep)
    return "\n".join(rows)


def load_pdf(file_path: str) -> list[Document]:
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            tables = page.extract_tables()
            if tables:
                text += "\n\n--- Table ---\n"
                for table in tables:
                    text += _table_to_markdown(table) + "\n"
            docs.append(
                Document(page_content=text, metadata={"page": i + 1, "source": file_path})
            )
    if not docs:
        raise ValueError(f"No text could be extracted from {file_path}")
    return docs


def split_pdf(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sigcomm16_cs2p.pdf"
    docs = load_pdf(path)
    splits = split_pdf(docs)
    print(f"Loaded pages: {len(docs)}")
    print(f"Total chunks:  {len(splits)}")
    print("\nFirst chunk:\n")
    print(splits[0].page_content[:500])
    print("\nMetadata:")
    print(splits[0].metadata)
