import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def check_resources():
    print("--- Environment Check (OpenAI Migration) ---")
    print(f"DB_HOST: {os.getenv('DB_HOST')}")
    print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")

    # 1. Check OpenAI API
    print("\n--- Checking OpenAI ---")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY is missing from environment variables.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            # Simple test to check API connectivity
            models = client.models.list()
            print("✅ OpenAI API Connectivity: OK")
            print(f"✅ API Key validated (can list models).")
        except Exception as e:
            print(f"❌ Error connecting to OpenAI: {e}")

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
        print(f"✅ DB Version: {cur.fetchone()[0]}")
        
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("✅ pgvector extension is already installed.")
        else:
            print("⚠️ pgvector extension is NOT installed. Attempting to enable...")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                print("✅ pgvector enabled successfully.")
            except Exception as ex:
                print(f"❌ Failed to enable pgvector: {ex}")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error connecting to DB: {e}")

if __name__ == "__main__":
    check_resources()
