import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Using text-embedding-3-small for production efficiency
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Intent Categories & Keywords for heuristic re-ranking
INTENTS = {
    "contact": ["phone", "email", "contact", "address", "location"],
    "services": ["services", "offerings", "solutions", "support", "bpo", "it", "recruitment", "staffing", "digital marketing"],
    "company": ["about", "mission", "vision", "company", "overview"]
}

def get_query_embedding(query):
    """
    Generate embedding for the user query via OpenAI API.
    Replaces the previous local Ollama dependency.
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def detect_intent(query):
    """Simple keyword-based intent detection."""
    query_lower = query.lower()
    detected = []
    for intent, keywords in INTENTS.items():
        if any(kw in query_lower for kw in keywords):
            detected.append(intent)
    return detected

def retrieve_top_chunks(query, top_k=5):
    """
    Retrieve top chunks with semantic search and heuristic re-ranking.
    Uses pgvector for similarity search in the PostgreSQL database.
    """
    embedding = get_query_embedding(query)
    if not embedding:
        return []

    intents = detect_intent(query)
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD
        )
        cur = conn.cursor()

        # SQL similarity search using pgvector
        # Note: Ensure the 'chunks' table in production has been re-indexed with OpenAI embeddings
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
            
            # Apply Boosting
            boost = 1.0
            
            # 1. Section Preference
            if "contact" in intents and chunk["section"] == "contact_information":
                boost += 0.2
            if "services" in intents and chunk["section"] in ["services", "manpower_supply"]:
                boost += 0.2
            
            # 2. Priority Boost
            if chunk["priority"] == "high":
                boost += 0.05
            
            # 3. Down-rank irrelevant policies
            if chunk["section"] == "policies" and not any(kw in query.lower() for kw in ["privacy", "policy", "legal"]):
                boost -= 0.1
            
            chunk["final_score"] = chunk["score"] * boost
            results.append(chunk)

        # Sort by final score and take top_k
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        print(f"Database error during retrieval: {e}")
        return []

if __name__ == "__main__":
    test_query = "How can I contact IAT Networks?"
    chunks = retrieve_top_chunks(test_query, top_k=3)
    for c in chunks:
        print(f"[{c['id']}] Score: {c['final_score']:.4f} | Section: {c['section']}")
        print(f"Text: {c['text'][:100]}...\n")
