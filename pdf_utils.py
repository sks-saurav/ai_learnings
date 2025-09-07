
import PyPDF2
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks.

    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk.
        overlap (int): The overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
