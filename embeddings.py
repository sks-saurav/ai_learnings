
from sentence_transformers import SentenceTransformer
from typing import List

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads a sentence-transformer model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        SentenceTransformer: The loaded sentence-transformer model.
    """
    return SentenceTransformer(model_name)

def generate_embeddings(chunks: List[str], model: SentenceTransformer) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks.

    Args:
        chunks (List[str]): A list of text chunks.
        model (SentenceTransformer): The sentence-transformer model to use.

    Returns:
        List[List[float]]: A list of embeddings.
    """
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings.tolist()
