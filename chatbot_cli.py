import sys
from retriever import retrieve_top_chunks
from generator import generate_answer

def run_chat():
    print("--- IAT Networks Front Desk Assistant (CLI Version) ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye! Have a professional day.")
            break
        
        if not user_input:
            continue
            
        print("\n(Retrieving knowledge...)")
        relevant_chunks = retrieve_top_chunks(user_input, top_k=3)
        
        # Internal debug info (won't be shown to end-user in production)
        if len(relevant_chunks) > 0:
            print(f"(Found {len(relevant_chunks)} relevant chunks. Top Score: {relevant_chunks[0]['final_score']:.4f})")
            for i, chunk in enumerate(relevant_chunks):
                print(f"  [{i+1}] {chunk['id']} ({chunk['section']}) - Sim: {chunk['score']:.4f}")
        else:
            print("(No relevant chunks found.)")
            
        print("\nAssistant is generating response...")
        answer = generate_answer(user_input, relevant_chunks)
        
        print(f"\nAssistant: {answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    try:
        run_chat()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
