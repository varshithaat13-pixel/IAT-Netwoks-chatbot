import os
import json
import psycopg2
import requests
import time
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load config
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# HuggingFace Config
HF_API_KEY = os.getenv('HF_API_KEY')
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Correct Router URL for HuggingFace Inference API
HF_EMBED_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"

BATCH_SIZE = 10 # Optimized batch size for ingestion

def get_embedding(text):
    """Generate embedding using HuggingFace Inference API with retry logic."""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    for attempt in range(5): # More retries for ingestion
        try:
            response = requests.post(
                HF_EMBED_URL,
                headers=headers,
                json={"inputs": text},
                timeout=20
            )
            
            if response.status_code == 200:
                embedding = response.json()
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        return embedding[0]
                    return embedding
                raise ValueError("Unexpected response format from HF")
            
            if response.status_code == 503:
                print(f"HF API 503: Model loading. Waiting 15s (Attempt {attempt+1})")
                time.sleep(15)
                continue
                
            if response.status_code == 429:
                print("HF API 429: Rate limit hit. Waiting 20s...")
                time.sleep(20)
                continue

            print(f"HF API Error: {response.status_code} - {response.text}")
            raise Exception(f"HF API Error {response.status_code}")
            
        except requests.exceptions.Timeout:
            print("HF API Timeout. Retrying...")
            continue
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise e
            
    raise Exception("Max retries exceeded for HF Embedding API")

def get_dim():
    """Detect embedding dimension for the selected model."""
    print(f"Validating embedding dimension for model {EMBEDDING_MODEL}...")
    emb = get_embedding("test")
    dim = len(emb)
    print(f"Detected dimension: {dim}")
    return dim

def setup_db(dim):
    """Setup PostgreSQL database with pgvector support."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    cur = conn.cursor()
    
    # Ensure pgvector is enabled
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Note: If migrating embedding models with different dimensions,
    # the table MUST be recreated.
    cur.execute(f"""
        DO $$ 
        BEGIN 
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks') THEN
                -- Check if dimension matches (384 for all-MiniLM-L6-v2)
                IF (SELECT atttypmod FROM pg_attribute 
                    WHERE attrelid = 'chunks'::regclass AND attname = 'embedding') != {dim} + 4 THEN
                    RAISE NOTICE 'Dimension mismatch detected. Dropping and recreating table.';
                    DROP TABLE chunks;
                END IF;
            END IF;
        END $$;
    """)

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
    """Run the ingestion pipeline: Read JSON -> Embed -> Upsert to DB."""
    try:
        with open('final_chunks.json', 'r', encoding='utf-16') as f:
            chunks = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
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
            "error": str(e),
            "notes": "Ensure database is reachable and HF API key is valid."
        }

    upsert_query = """
    INSERT INTO chunks (id, text, embedding, section, sub_section, intent, keywords, priority, source)
    VALUES %s
    ON CONFLICT (id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        text = EXCLUDED.text,
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

    summary = {
        "status": "success" if not failed_chunks else "partial_success",
        "total_chunks": total_chunks,
        "inserted_chunks": inserted_count,
        "failed_chunks": len(failed_chunks),
        "embedding_model": EMBEDDING_MODEL,
        "database": DB_NAME
    }
    return summary

if __name__ == "__main__":
    print(f"--- IAT Networks Knowledge Ingestion (HF Router Fix) ---")
    result_summary = ingest()
    print("\n--- INGESTION SUMMARY ---")
    print(json.dumps(result_summary, indent=2))
