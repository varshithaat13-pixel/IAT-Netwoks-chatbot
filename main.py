import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from retriever import retrieve_top_chunks
from generator import generate_answer

app = FastAPI(
    title="IAT Networks Front Desk AI",
    description="Ultra-lightweight RAG Chatbot optimized for Render Free Tier (API-based)",
    version="1.2.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://iat-netwoks-chatbot-1.onrender.com",
        "https://iat-netwoks-chatbot.onrender.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class SourceChunk(BaseModel):
    id: str
    section: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

@app.get("/")
def read_root():
    return {"message": "IAT Networks Front Desk Assistant API is online."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for the IAT Networks RAG Chatbot.
    Accepts a user query, retrieves relevant information, and generates a response via Groq.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Retrieval & Re-ranking (Local Sentence-Transformers)
        relevant_chunks = retrieve_top_chunks(request.query, top_k=request.top_k)
        
        # 2. Generation (Groq Llama-3)
        answer = generate_answer(request.query, relevant_chunks)
        
        # 3. Format Response
        sources = [
            SourceChunk(id=c['id'], section=c['section'], score=c['score']) 
            for c in relevant_chunks
        ]
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )

    except Exception as e:
        print(f"CRITICAL SERVER ERROR: {e}")
        return ChatResponse(
            answer="I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again in a few seconds.",
            sources=[]
        )

# CLI Entry point for production/local dev
if __name__ == "__main__":
    # Use PORT environment variable for Render compatibility
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
