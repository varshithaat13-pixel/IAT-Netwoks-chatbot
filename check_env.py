import os
import psycopg2
from dotenv import load_dotenv
import requests

load_dotenv()

def check_resources():
    print("--- Environment Check ---")
    print(f"DB_HOST: {os.getenv('DB_HOST')}")
    print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")
    print(f"OLLAMA_EMBED_MODEL: {os.getenv('OLLAMA_EMBED_MODEL')}")

    # 1. Check Ollama
    print("\n--- Checking Ollama ---")
    try:
        response = requests.get(f"{os.getenv('OLLAMA_BASE_URL')}/api/tags")
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"Available models: {models}")
            if any(os.getenv('OLLAMA_EMBED_MODEL').split(':')[0] in m for m in models):
                print("Model found!")
            else:
                print("Warning: Model not found in tags, will try to pull if needed or fail during embedding.")
        else:
            print(f"Failed to reach Ollama: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")

    # 2. Check Database and pgvector
    print("\n--- Checking Database ---")
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        print(f"DB Version: {cur.fetchone()[0]}")
        
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("pgvector extension is already installed.")
        else:
            print("pgvector extension is NOT installed. Attempting to enable (requires superuser/owner)...")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                print("pgvector enabled successfully.")
            except Exception as ex:
                print(f"Failed to enable pgvector: {ex}")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error connecting to DB: {e}")

if __name__ == "__main__":
    check_resources()
