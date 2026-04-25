import json
from retriever import retrieve_top_chunks
from generator import generate_answer

def run_test(query):
    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"{'='*50}")
    
    # 1. Retrieval
    chunks = retrieve_top_chunks(query, top_k=3)
    print("\n[RETRIEVAL RESULTS]")
    for i, c in enumerate(chunks):
        print(f"{i+1}. ID: {c['id']} | Section: {c['section']} | Score: {c['final_score']:.4f}")
    
    # 2. Generation
    print("\n[GENERATED RESPONSE]")
    answer = generate_answer(query, chunks)
    print(answer)
    print(f"\n{'='*50}\n")

if __name__ == "__main__":
    queries = [
        "How can I contact the HR team?",
        "What BPO services do you provide?",
        "What is your privacy policy regarding data?",
        "Who is the president of France?" # Outside Context
    ]
    
    for q in queries:
        run_test(q)
