
from llama_cpp import Llama
from typing import List, Dict, Any

def load_llama_model(model_path: str) -> Llama:
    """
    Loads a Llama model.

    Args:
        model_path (str): The path to the Llama model.

    Returns:
        Llama: The loaded Llama model.
    """
    return Llama(model_path=model_path)

def create_prompt(question: str, context: List[Dict[str, Any]]) -> str:
    """
    Creates a prompt for the Llama model.

    Args:
        question (str): The user's question.
        context (List[Dict[str, Any]]): The context retrieved from Qdrant.

    Returns:
        str: The prompt for the Llama model.
    """
    context_str = "\n".join([item.payload["text"] for item in context])
    prompt = f"""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context_str}

    Question: {question}
    Helpful Answer:
    """
    return prompt

def generate_answer(model: Llama, prompt: str) -> str:
    """
    Generates an answer using the Llama model.

    Args:
        model (Llama): The Llama model.
        prompt (str): The prompt for the Llama model.

    Returns:
        str: The generated answer.
    """
    output = model(prompt, max_tokens=256, echo=False)
    return output["choices"][0]["text"]
