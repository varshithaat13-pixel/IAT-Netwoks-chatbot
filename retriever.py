import os
import psycopg2
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# HuggingFace Configuration
HF_API_KEY = os.getenv("HF_API_KEY")
# Using the updated Router URL for HuggingFace Inference
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBED_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"

# Intent Categories & Keywords for heuristic re-ranking
INTENTS = {
    "contact": ["phone", "email", "contact", "address", "location"],
    "services": ["services", "offerings", "solutions", "support", "bpo", "it", "recruitment", "staffing", "digital marketing"],
    "company": ["about", "mission", "vision", "company", "overview"]
}

def get_query_embedding(text: str):
    """
    Generate embedding for the user query via HuggingFace Inference API.
    Uses the modern router.huggingface.co endpoint for stability.
    """
    if not HF_API_KEY or "your_" in HF_API_KEY:
        print("Error: Invalid HF_API_KEY found in environment.")
        return None

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # Retry logic for 503 (loading) and 429 (rate limits)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Task: Feature Extraction (Embeddings)
            response = requests.post(
                HF_EMBED_URL,
                headers=headers,
                json={"inputs": text},
                timeout=10 # Strict 10s timeout for Render stability
            )
            
            # 1. Handle Success
            if response.status_code == 200:
                embedding = response.json()
                # The API returns a list (vector). Handle potential nesting.
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        return embedding[0] # Nested list case
                    return embedding
                return None
            
            # 2. Handle 404 (Wrong Endpoint)
            if response.status_code == 404:
                print(f"HF API Error 404: Endpoint not found at {HF_EMBED_URL}")
                return None

            # 3. Handle 503 (Model Loading / Cold Start)
            if response.status_code == 503:
                wait_time = 10 if attempt == 0 else 20
                print(f"HF API 503: Model is loading. Waiting {wait_time}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
                
            # 4. Handle 429 (Rate Limit)
            if response.status_code == 429:
                print(f"HF API 429: Rate limit hit. Waiting 15s (Attempt {attempt+1}/{max_retries})")
                time.sleep(15)
                continue

            # 5. Handle Other Errors
            print(f"HF API Error: {response.status_code} - {response.text}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"HF API Timeout: Request took longer than 10s (Attempt {attempt+1}/{max_retries})")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"Error during HF API call: {e}")
            return None
            
    print("HF API Error: Max retries exceeded.")
    return None

def detect_intent(query):
    """Simple keyword-based intent detection."""
    query_lower = query.lower()
    detected = []
    for intent, keywords in INTENTS.items():
        if any(kw in query_lower for kw in keywords):
            detected.append(intent)
    return detected

def retrieve_top_chunks(query, top_k=3):
    """
    Retrieve top chunks with semantic search and heuristic re-ranking.
    """
    try:
        embedding = get_query_embedding(query)
    except Exception as e:
        print("Critical Embedding Error:", e)
        return []

    if not embedding:
        return []

    intents = detect_intent(query)
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD
        )
        cur = conn.cursor()

        # SQL similarity search using pgvector (Cosine Similarity)
        # Note: chunks.embedding must match the 384 dimensions of all-MiniLM-L6-v2
        search_query = """
        SELECT id, text, section, sub_section, priority, 
               1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 10;
        """
        cur.execute(search_query, (embedding, embedding))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Re-ranking logic
        results = []
        for row in rows:
            chunk = {
                "id": row[0],
                "text": row[1],
                "section": row[2],
                "sub_section": row[3],
                "priority": row[4],
                "score": float(row[5])
            }
            
            # Simple boosting
            boost = 1.0
            if "contact" in intents and chunk["section"] == "contact_information": boost += 0.2
            if "services" in intents and chunk["section"] == "services": boost += 0.2
            
            chunk["final_score"] = chunk["score"] * boost
            results.append(chunk)

        # Sort by final score and take top_k
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        print(f"Database error during retrieval: {e}")
        return []

if __name__ == "__main__":
    # Test case
    test_query = "What is the phone number of IAT Networks?"
    print(f"Testing retrieval for: {test_query}")
    chunks = retrieve_top_chunks(test_query, top_k=2)
    if chunks:
        for c in chunks:
            print(f"[{c['id']}] Score: {c['final_score']:.4f} | Text: {c['text'][:100]}...")
    else:
        print("No chunks retrieved. Ensure database is ingested with 384-dim embeddings.")
