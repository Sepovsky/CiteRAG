from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_pdf(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150
    )
    
    return text_splitter.split_documents(docs)

# if __name__ == "__main__":
#     docs = load_pdf("data/sigcomm16_cs2p.pdf")
#     print(f"Loaded {len(docs)} pages")
#     print(docs[0].page_content[:500])
#     print(docs[0].metadata)

if __name__ == "__main__":
    docs = load_pdf("data/sigcomm16_cs2p.pdf")
    splits = split_pdf(docs)

    print(f"Loaded pages: {len(docs)}")
    print(f"Total chunks: {len(splits)}")
    print("\nFirst chunk:\n")
    print(splits[0].page_content[:500])
    print("\nMetadata:")
    print(splits[0].metadata)