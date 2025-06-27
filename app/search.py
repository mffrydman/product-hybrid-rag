from datetime import datetime
from database.vector_store import VectorStore
from timescale_vector import client
from services.synthesizer import Synthesizer
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")

vec = VectorStore()

query = "I want an elegant and comfortable everyday black tote bag or backpack that i can use on the street or at work"

# --------------------------------------------------------------
# Semantic search
# --------------------------------------------------------------

semantic_results = vec.semantic_search(query=query, limit=5)

response = Synthesizer.generate_response(question=query, context=semantic_results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nConfiability: {response.confiability}")

# --------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------

metadata_filter = {"gender": "F"}

semantic_results = vec.semantic_search(
    query=query, limit=5, metadata_filter=metadata_filter
)

response = Synthesizer.generate_response(question=query, context=semantic_results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nConfiability: {response.confiability}")

# --------------------------------------------------------------
# Filtering with predicates
# --------------------------------------------------------------

predicates = (
    client.Predicates("gender", "==", "F")
    & client.Predicates("price", "<=", 99.99)
    & client.Predicates("category-2", "==", "backpack")
)
results = vec.semantic_search(query, limit=3, predicates=predicates)

response = Synthesizer.generate_response(question=query, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nConfiability: {response.confiability}")

# --------------------------------------------------------------
# Filtering with date ranges
# --------------------------------------------------------------

time_range = (datetime(2025, 5, 1), datetime(2025, 5, 30))
results = vec.semantic_search(query, limit=3, time_range=time_range)


# --------------------------------------------------------------
# Keyword search
# --------------------------------------------------------------

keyword_results = vec.keyword_search(query=query, limit=5)

# --------------------------------------------------------------
# Hybrid search -> Keyword + Semantic
# --------------------------------------------------------------

hybrid_results = vec.hybrid_search(query=query, keyword_k=10, semantic_k=10)

# --------------------------------------------------------------
# Reranking step with Cohere API
# --------------------------------------------------------------

reranked_results = vec.hybrid_search(
    query=query, keyword_k=10, semantic_k=10, rerank=True, top_n=5
)

# --------------------------------------------------------------
# Response
# --------------------------------------------------------------

response = Synthesizer.generate_response(question=query, context=reranked_results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nConfiability: {response.confiability}")
