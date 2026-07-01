import os
import pdfplumber
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# Подключение к единой базе данных
DATABASE_URL = 'postgresql://user:pass@localhost:5433/intel_gateway'
engine = create_engine(DATABASE_URL)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_into_chunks(text, chunk_size=800, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_instruction():
    pdf_path = "instruction.pdf"  # Путь к файлу инструкции
    if not os.path.exists(pdf_path):
        print("Файл instruction.pdf не найден в папке!")
        return

    print("Извлечение текста из PDF...")
    text = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(text, 800, 200)
    print(f"Получено {len(chunks)} чанков.")

    print("Вычисление эмбеддингов и сохранение в БД...")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS normative_chunks (
                id SERIAL PRIMARY KEY,
                section VARCHAR(100),
                content TEXT,
                embedding VECTOR(384)
            );
        """))
        conn.commit()

        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk).tolist()
            emb_str = '[' + ','.join(str(x) for x in emb) + ']'
            conn.execute(text("""
                INSERT INTO normative_chunks (section, content, embedding)
                VALUES (:section, :content, :embedding)
            """), {"section": f"Чанк {i+1}", "content": chunk, "embedding": emb_str})
        conn.commit()
    print("Загрузка завершена!")

if __name__ == "__main__":
    load_instruction()