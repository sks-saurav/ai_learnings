'''
Semantic Search Engine using Qdrant and Sentence Transformers

pip install sentence-transformers qdrant-client
docker run -p 6333:6333 qdrant/qdrant
'''

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# 1. Generate embeddings
def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for a list of texts using the provided model.
    """
    return np.array(model.encode(texts, show_progress_bar=True))

# 2. Create Qdrant collection
def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 384):
    """
    Create a Qdrant collection for storing embeddings.
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

# 3. Insert documents with embeddings and metadata
def insert_documents(client: QdrantClient, collection_name: str, texts: List[str], embeddings: np.ndarray, metadatas: List[dict]):
    """
    Insert documents and their embeddings into Qdrant.
    """
    points = [
        PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload={**metadatas[idx], "text": texts[idx]}
        )
        for idx in range(len(texts))
    ]
    client.upsert(collection_name=collection_name, points=points)

# 4. Semantic search
def search_documents(client: QdrantClient, collection_name: str, query: str, model: SentenceTransformer, top_k: int = 3):
    """
    Search for documents semantically similar to the query.
    """
    query_embedding = model.encode(query)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    return search_result

# 5. Print results neatly
def print_results(results):
    print("\nTop Results:")
    for res in results:
        payload = res.payload
        print(f"Score: {res.score:.4f}")
        print(f"Text: {payload.get('text')}")
        print(f"ID: {payload.get('id', 'N/A')}, Author: {payload.get('author', 'N/A')}, Date: {payload.get('date', 'N/A')}")
        print("-" * 40)

if __name__ == "__main__":
    # Connect to local Qdrant
    client = QdrantClient(host="localhost", port=6333)

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Sample documents and metadata
    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps computers understand human language.",
        "Deep learning is a subset of machine learning.",
        "Artificial intelligence is transforming industries.",
        "Machine learning enables predictive analytics.",
        "Natural language processing powers chatbots.",
        "Deep learning improves image recognition.",
        "Reinforcement learning optimizes decision making.",
        "Transfer learning accelerates model training.",
        "Neural networks mimic brain functions.",
        "Supervised learning uses labeled data.",
        "Unsupervised learning finds hidden patterns.",
        "Data science drives business insights."
    ]
    metadatas = [
        {"id": 1, "author": "Alice", "date": "2025-09-01"},
        {"id": 2, "author": "Bob", "date": "2025-09-02"},
        {"id": 3, "author": "Carol", "date": "2025-09-03"},
        {"id": 4, "author": "Dave", "date": "2025-09-04"},
        {'id': 5, 'author': 'Carol', 'date': '2025-01-01'},
        {'id': 6, 'author': 'Heidi', 'date': '2025-01-02'},
        {'id': 7, 'author': 'Heidi', 'date': '2025-01-03'},
        {'id': 8, 'author': 'Bob', 'date': '2025-01-04'},
        {'id': 9, 'author': 'Eve', 'date': '2025-01-05'},
        {'id': 10, 'author': 'Frank', 'date': '2025-01-06'},
        {'id': 11, 'author': 'Heidi', 'date': '2025-01-07'},
        {'id': 12, 'author': 'Eve', 'date': '2025-01-08'},
        {'id': 13, 'author': 'Dave', 'date': '2025-01-09'},
        {'id': 14, 'author': 'Carol', 'date': '2025-01-10'},
    ]

    collection_name = "semantic_search_demo"

    # Step 1: Generate embeddings
    embeddings = generate_embeddings(texts, model)

    # Step 2: Create collection
    create_collection(client, collection_name)

    # Step 3: Insert documents
    insert_documents(client, collection_name, texts, embeddings, metadatas)

    # Step 4: Semantic search
    query = "How do computers understand language?"
    results = search_documents(client, collection_name, query, model, top_k=3)

    # Step 5: Print results
    print_results(results)