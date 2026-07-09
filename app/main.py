"""Async interactive CLI for CiteRAG with streaming output."""

import asyncio
from typing import List

from langchain_core.documents import Document
from qa import stream_answer


async def main():
    document_path = "data/sigcomm16_cs2p.pdf"

    print("CiteRAG — Citation-Grounded RAG for Document QA")
    print("Answers include page-level citations. Enter 'q' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "q":
            print("Goodbye!")
            break
        if not question:
            continue

        print("\nAnswer:\n")
        answer_parts: List[str] = []
        docs: List[Document] = []

        async for item in stream_answer(document_path, question):
            if isinstance(item, tuple) and item[0] == "__docs__":
                docs = item[1]
            else:
                print(item, end="", flush=True)
                answer_parts.append(item)

        pages = sorted({doc.metadata.get("page", "N/A") for doc in docs})
        if pages:
            print("\n\nRetrieved source pages:", pages)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
