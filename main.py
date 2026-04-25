import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from retriever import retrieve_top_chunks
from generator import generate_answer

app = FastAPI(
    title="IAT Networks Front Desk AI",
    description="RAG-based Chatbot API for IAT Networks Internal Knowledge Base",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local development
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
    Accepts a user query, retrieves relevant information, and generates a response.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Retrieval & Re-ranking
        relevant_chunks = retrieve_top_chunks(request.query, top_k=request.top_k)
        
        # 2. Generation
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
        # Logging here in a real production environment
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# CLI Entry point for local dev
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
