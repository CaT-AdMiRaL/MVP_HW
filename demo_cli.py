from rag_utils import Retriever, Generator, filter_chunks


def main():
    retriever = Retriever()
    generator = Generator()

    print("RAG по стратегии развития ИИ до 2030 года")
    print("Для выхода введи: exit\n")

    while True:
        question = input("Вопрос: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        found = retriever.search(question, top_k=4)
        good = filter_chunks(found, min_score=0.35)

        print("\nНайденные фрагменты:")
        for chunk in found:
            preview = chunk["text"][:170].replace("\n", " ")
            print(f"- стр. {chunk['page']} | score={chunk['score']:.4f} | {preview}...")

        if not good:
            answer = "В документе это не найдено."
        else:
            answer = generator.answer(question, good)

        print("\nОтвет:")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()