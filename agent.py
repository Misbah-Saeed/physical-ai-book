import os
import random
from dotenv import load_dotenv
from main import retrieve  # reuse RAG retrieval logic from main.py

load_dotenv()

# --- SIMPLE AGENT ---
def how_many_jokes():
    """Return a random number for jokes"""
    return random.randint(1, 10)

def answer_query(user_input):
    """
    Agent answers user queries using RAG retrieval and simple logic
    """
    # Step 1: Retrieve top results from RAG
    results = retrieve(user_input, top_k=5)
    
    # Step 2: Basic logic for jokes
    if "joke" in user_input.lower():
        n = how_many_jokes()
        return f"Here are {n} random jokes (not real content, just a placeholder)."
    
    # Step 3: Return retrieved results as answer
    return "\n".join([f"{i+1}. {text[:200]}..." for i, text in enumerate(results)])

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    while True:
        query = input("\nAsk the agent: ")
        answer = answer_query(query)
        print("\nAgent answer:")
        print(answer)
