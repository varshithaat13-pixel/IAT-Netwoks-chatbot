import os
import json
import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load config
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL')

BATCH_SIZE = 5

def get_embedding(text):
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]

def get_dim():
    print(f"Validating embedding dimension for model {OLLAMA_EMBED_MODEL}...")
    emb = get_embedding("test")
    dim = len(emb)
    print(f"Detected dimension: {dim}")
    return dim

def setup_db(dim):
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    cur = conn.cursor()
    
    # Ensure pgvector is enabled
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create table
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        embedding VECTOR({dim}),
        section TEXT,
        sub_section TEXT,
        intent TEXT,
        keywords TEXT[],
        priority TEXT,
        source TEXT
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    return conn, cur

def ingest():
    try:
        # PowerShell redirects often use utf-16
        with open('final_chunks.json', 'r', encoding='utf-16') as f:
            chunks = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Fallback to utf-8
        with open('final_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    
    total_chunks = len(chunks)
    inserted_count = 0
    failed_chunks = []
    
    try:
        dim = get_dim()
        conn, cur = setup_db(dim)
    except Exception as e:
        return {
            "status": "failure",
            "total_chunks": total_chunks,
            "inserted_chunks": 0,
            "failed_chunks": total_chunks,
            "embedding_model": OLLAMA_EMBED_MODEL,
            "database": DB_NAME,
            "table": "chunks",
            "notes": [f"Setup failed: {str(e)}"]
        }

    upsert_query = """
    INSERT INTO chunks (id, text, embedding, section, sub_section, intent, keywords, priority, source)
    VALUES %s
    ON CONFLICT (id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        section = EXCLUDED.section,
        sub_section = EXCLUDED.sub_section,
        intent = EXCLUDED.intent,
        keywords = EXCLUDED.keywords,
        priority = EXCLUDED.priority,
        source = EXCLUDED.source;
    """

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_data = []
        
        for chunk in batch:
            try:
                embedding = get_embedding(chunk['text'])
                batch_data.append((
                    chunk['id'],
                    chunk['text'],
                    embedding,
                    chunk.get('section'),
                    chunk.get('sub_section'),
                    chunk.get('intent'),
                    chunk.get('keywords', []),
                    chunk.get('priority'),
                    chunk.get('source')
                ))
            except Exception as e:
                failed_chunks.append({"id": chunk['id'], "error": str(e)})
                print(f"FAILED chunk {chunk['id']}: {e}")
        
        if batch_data:
            try:
                execute_values(cur, upsert_query, batch_data)
                conn.commit()
                inserted_count += len(batch_data)
                print(f"Committed batch of {len(batch_data)} chunks ({inserted_count}/{total_chunks})")
            except Exception as e:
                conn.rollback()
                failed_chunks.append({"batch_range": f"{i}-{i+BATCH_SIZE}", "error": f"DB Error: {str(e)}"})
                print(f"DB BATCH FAILED: {e}")

    cur.close()
    conn.close()

    status = "success" if not failed_chunks else ("partial_success" if inserted_count > 0 else "failure")
    
    summary = {
        "status": status,
        "total_chunks": total_chunks,
        "inserted_chunks": inserted_count,
        "failed_chunks": len(failed_chunks),
        "embedding_model": OLLAMA_EMBED_MODEL,
        "database": DB_NAME,
        "table": "chunks",
        "notes": [f"Errors: {failed_chunks}"] if failed_chunks else []
    }
    return summary

def retrieval_test(query, top_k=3):
    print(f"\n--- Retrieval Test for query: '{query}' ---")
    try:
        emb = get_embedding(query)
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD
        )
        cur = conn.cursor()
        
        # Simple cosine similarity (using <=> for cosine distance, so smaller is closer)
        test_query = """
        SELECT id, text, section, 1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        cur.execute(test_query, (emb, emb, top_k))
        results = cur.fetchall()
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "id": r[0],
                "text": r[1][:100] + "...",
                "section": r[2],
                "similarity": float(r[3])
            })
        
        cur.close()
        conn.close()
        return formatted_results
    except Exception as e:
        return {"error": f"Retrieval failed: {str(e)}"}

if __name__ == "__main__":
    result_summary = ingest()
    print("\n--- INGESTION SUMMARY ---")
    print(json.dumps(result_summary, indent=2))
    
    # Short delay to ensure commit visibility or just proceed
    test_query = "What are the contact details for IAT Networks?"
    retrieval_results = retrieval_test(test_query)
    print("\n--- RETRIEVAL RESULTS ---")
    print(json.dumps(retrieval_results, indent=2))
