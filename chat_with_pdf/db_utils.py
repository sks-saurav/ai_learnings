
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any

def get_qdrant_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """
    Creates a Qdrant client.

    Args:
        host (str): The host of the Qdrant instance.
        port (int): The port of the Qdrant instance.

    Returns:
        QdrantClient: The Qdrant client.
    """
    return QdrantClient(host=host, port=port)

def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Creates a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client.
        collection_name (str): The name of the collection.
        vector_size (int): The size of the vectors.
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

def upload_embeddings(client: QdrantClient, collection_name: str, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
    """
    Uploads embeddings to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client.
        collection_name (str): The name of the collection.
        embeddings (List[List[float]]): A list of embeddings.
        metadata (List[Dict[str, Any]]): A list of metadata dictionaries.
    """
    client.upload_collection(
        collection_name=collection_name,
        vectors=embeddings,
        payload=metadata,
        ids=None,  # auto-assign
        batch_size=256
    )

def search_qdrant(client: QdrantClient, collection_name: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches for similar vectors in a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client.
        collection_name (str): The name of the collection.
        query_embedding (List[float]): The query embedding.
        top_k (int): The number of similar vectors to return.

    Returns:
        List[Dict[str, Any]]: A list of search results.
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
    )
    return search_result
