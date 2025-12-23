import os
import cohere
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "physical_ai_book"

cohere_client = cohere.Client(COHERE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- EMBEDDING ---
def get_embedding(text):
    res = cohere_client.embed(
        model="embed-english-v3.0",
        texts=[text],
        input_type="search_query"   # 🔴 yeh line add karo
    )
    return res.embeddings[0]
# --- RAG RETRIEVAL ---
def retrieve(query, top_k=5):
    query_vector = get_embedding(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    return [point.payload["text"] for point in results.points]

# --- CHAT LOOP ---
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk me something: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        answers = retrieve(user_input)
        print("\nTop results:")
        for i, a in enumerate(answers, 1):
            print(f"{i}. {a[:200]}...")
