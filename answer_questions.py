import pandas as pd
from rag_utils import Retriever, Generator, filter_chunks

INPUT_XLSX = "data/test_set.xlsx"
OUTPUT_XLSX = "test_set_Чирухина_Алина.xlsx"   # замени на свои ФИО


def detect_question_column(df: pd.DataFrame) -> str:
    candidates = ["question", "questions", "query", "prompt", "вопрос"]
    for col in df.columns:
        if str(col).strip().lower() in candidates:
            return col
    raise ValueError(f"Не найдена колонка с вопросами. Колонки: {list(df.columns)}")


def main():
    df = pd.read_excel(INPUT_XLSX)

    question_col = detect_question_column(df)

    if "answer" not in df.columns:
        df["answer"] = ""
    else:
        df["answer"] = df["answer"].fillna("")

    retriever = Retriever()
    generator = Generator()

    total = len(df)

    for idx, row in df.iterrows():
        question = str(row[question_col]).strip()

        if not question:
            df.at[idx, "answer"] = ""
            continue

        print(f"\n[{idx + 1}/{total}] Вопрос: {question}")

        found = retriever.search(question, top_k=4)
        good = filter_chunks(found, min_score=0.35)

        print("Найденные фрагменты:")
        for chunk in found:
            preview = chunk["text"][:120].replace("\n", " ")
            print(f"  - стр. {chunk['page']} | score={chunk['score']:.4f} | {preview}...")

        if not good:
            answer = "В документе это не найдено."
        else:
            answer = generator.answer(question, good)

        df.at[idx, "answer"] = answer
        print(f"Ответ: {answer}")

    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nГотово. Сохранено: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()