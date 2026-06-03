from qa import ask_document


def main():
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

        answer, docs = ask_document(document_path, question)
        print("\nAnswer:\n")
        print(answer)

        pages = sorted({doc.metadata.get("page", "N/A") for doc in docs})
        print("\nRetrieved source pages:", pages)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
