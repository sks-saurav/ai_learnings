
import os
from pdf_utils import extract_text_from_pdf, split_text_into_chunks
from embeddings import get_embedding_model, generate_embeddings
from db_utils import get_qdrant_client, create_collection, upload_embeddings, search_qdrant
from llm import load_llama_model, create_prompt, generate_answer

def main():
    """
    Main function for the Chat with PDF application.
    """
    # Configuration
    PDF_PATH = "path/to/your/document.pdf"  # Replace with your PDF path
    COLLECTION_NAME = "my_pdf_collection"
    LLAMA_MODEL_PATH = "path/to/your/llama/model.gguf"  # Replace with your Llama model path

    # --- 1. Process PDF ---
    print("Processing PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    chunks = split_text_into_chunks(text)
    print(f"PDF processed into {len(chunks)} chunks.")

    # --- 2. Generate and Upload Embeddings ---
    print("Generating embeddings...")
    embedding_model = get_embedding_model()
    embeddings = generate_embeddings(chunks, embedding_model)
    vector_size = len(embeddings[0])
    print("Embeddings generated.")

    print("Connecting to Qdrant and uploading embeddings...")
    qdrant_client = get_qdrant_client()
    create_collection(qdrant_client, COLLECTION_NAME, vector_size)
    metadata = [{"text": chunk} for chunk in chunks]
    upload_embeddings(qdrant_client, COLLECTION_NAME, embeddings, metadata)
    print("Embeddings uploaded to Qdrant.")

    # --- 3. Load Llama Model ---
    print("Loading Llama model...")
    llama_model = load_llama_model(LLAMA_MODEL_PATH)
    print("Llama model loaded.")

    # --- 4. Ask Questions ---
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        # Embed the question
        query_embedding = generate_embeddings([question], embedding_model)[0]

        # Search for context
        context = search_qdrant(qdrant_client, COLLECTION_NAME, query_embedding)

        # Generate answer
        prompt = create_prompt(question, context)
        answer = generate_answer(llama_model, prompt)

        print("\nAnswer:")
        print(answer)
        print("\n---\n")

if __name__ == "__main__":
    # Note: You need to have a Qdrant instance running.
    # You can use Docker: docker run -p 6333:6333 qdrant/qdrant
    # You also need to download a Llama model in GGUF format.
    main()
