import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# Подключение к единой базе PostgreSQL с pgvector (порт 5433)
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost:5433/intel_gateway')
engine = create_engine(DATABASE_URL)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model

def find_similar_normative(query_text, top_k=3):
    """
    Выполняет семантический поиск по нормативной базе (Инструкция Минцифры).
    Возвращает top_k наиболее релевантных фрагментов в виде строки.
    """
    if not query_text:
        return ""

    model = get_model()
    query_emb = model.encode(query_text).tolist()
    emb_str = '[' + ','.join(str(x) for x in query_emb) + ']'

    with engine.connect() as conn:
        sql = text("""
            SELECT content, section, (embedding <-> CAST(:emb AS vector)) AS distance
            FROM normative_chunks
            ORDER BY embedding <-> CAST(:emb AS vector)
            LIMIT :top_k
        """)
        params = {"emb": emb_str, "top_k": top_k}
        result = conn.execute(sql, params).fetchall()

    chunks = [row.content for row in result]
    return "\n---\n".join(chunks)