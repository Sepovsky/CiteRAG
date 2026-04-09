from qa import ask_pdf

def main():
    pdf_path = "data/sigcomm16_cs2p.pdf"

    print("Welcome to the PDF QA system!")
    print("Enter 'q' to quit")

    while True:
        question = input("Enter your question: ")
        if question.lower() == 'q':
            print("Goodbye!")
            break

        answer, docs = ask_pdf(pdf_path, question)
        print("\nAnswer:\n")
        print(answer)

        print("\nSources:")
        pages = sorted(set(doc.metadata.get("page", "N/A") for doc in docs))
        print(pages)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()