import os
from rag_utils import (
    extract_pages_from_pdf,
    split_into_chunks,
    ollama_embed,
    save_chunks,
    save_faiss_index,
)

PDF_PATH = "data/strategy_ai_2030.pdf"


def main():
    os.makedirs("storage", exist_ok=True)

    print("1. Читаю PDF...")
    pages = extract_pages_from_pdf(PDF_PATH)
    print(f"Страниц: {len(pages)}")

    print("2. Режу на чанки...")
    chunks = split_into_chunks(pages, chunk_size=900, overlap=150)
    print(f"Чанков: {len(chunks)}")

    print("3. Делаю эмбеддинги через Ollama...")
    vectors = ollama_embed([chunk["text"] for chunk in chunks])

    print("4. Сохраняю индекс...")
    save_chunks(chunks)
    save_faiss_index(vectors)

    print("Готово.")


if __name__ == "__main__":
    main()