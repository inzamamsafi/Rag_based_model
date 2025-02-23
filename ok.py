import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import chromadb
from chromadb.config import Settings
import requests
from PIL import Image
import io
import os  # For environment variables
import google.generativeai as genai  # Import the google-generativeai library
import time  # Import the time module

# Initialize models
text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(persist_directory="./chromadb"))
# chroma_client = chromadb.Client(Settings(persist_directory=os.path.expanduser("~/chromadb"))) # You can use this for home directory if needed

# Gemini API Configuration using google-generativeai
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Get API key from environment variable

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set. Please set it before running.")
    exit()

genai.configure(api_key=GEMINI_API_KEY) # Configure Gemini API with the key

# Gemini Model Configuration (You can adjust these settings)
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
}

gemini_model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-thinking-exp-01-21", # Using "gemini-pro" model, you can change to "gemini-pro-vision" if needed for multimodal input later
  generation_config=generation_config,
  system_instruction="Answer the question from the context provided by the user. Be concise and informative.", # Improved system instruction
)

chat_session = gemini_model.start_chat() # Start a chat session globally, we'll reuse it


# Function to extract text and images from PDF (Correct - No changes needed)
def extract_pdf_data(pdf_path):
    doc = fitz.open(pdf_path)
    data = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        images = page.get_images(full=True)
        image_data = []
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data.append(base_image["image"])
        data.append({
            'page_num': page_num,
            'text': text,
            'images': image_data
        })
    return data

# Function to get text embedding (Correct - No changes needed)
def get_text_embedding(text):
    return text_model.encode(text)

# Function to get image embedding (Correct - No changes needed)
def get_image_embedding(image_data):
    image = Image.open(io.BytesIO(image_data))
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs.detach().numpy().flatten()

# Modified function to index data into separate ChromaDB collections with 2000 character chunks (Correct - No changes needed)
def index_data(data):
    print("--- Indexing data process STARTED ---") # ADDED: Start of indexing print
    start_time = time.time() # ADDED: Start time measurement

    text_collection = chroma_client.get_or_create_collection(name="pdf_text_data") # Separate text collection
    image_collection = chroma_client.get_or_create_collection(name="pdf_image_data") # Separate image collection

    for i, d in enumerate(data):
        # Index Text Data
        text_embedding = get_text_embedding(d['text'])
        text_chunks = [d['text'][i:i+2000] for i in range(0, len(d['text']), 2000)] # Chunk size 2000
        for chunk_index, text_chunk in enumerate(text_chunks):
            text_embedding_chunk = get_text_embedding(text_chunk)
            text_collection.add( # Add to text_collection
                ids=[f"text_{i}_{chunk_index}"],
                embeddings=[text_embedding_chunk.tolist()],
                metadatas=[{'page_num': d['page_num'], 'type': 'text', 'chunk_index': chunk_index}],
                documents=[text_chunk]
            )

        # Index Image Data
        for j, img in enumerate(d['images']):
            image_embedding = get_image_embedding(img)
            image_collection.add( # Add to image_collection
                ids=[f"image_{i}_{j}"],
                embeddings=[image_embedding.tolist()],
                metadatas=[{'page_num': d['page_num'], 'type': 'image', 'image_index': j}]
            )
    end_time = time.time() # ADDED: End time measurement
    indexing_time = end_time - start_time # ADDED: Calculate indexing time
    print(f"--- Indexing data process FINISHED --- Time taken: {indexing_time:.2f} seconds") # ADDED: End of indexing print with time
    return {"text_collection": text_collection, "image_collection": image_collection} # Return both collections

# Corrected function to query ChromaDB - now takes collection as argument (Correct - No changes needed)
def query_chromadb(collection, query_text, top_k=3): # top_k = number of results to retrieve
    query_embedding = get_text_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], # Ensure query embedding is a list
        n_results=top_k,
        include=["metadatas", "documents"] # Include metadata and documents in results
    )
    return results

# Modified function to get answer from Gemini API using google.generativeai (Correct - No changes needed)
def get_gemini_answer(query, context_chunks):
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not set."

    context_string = "\n\n".join(context_chunks) # Combine context chunks into a single string

    prompt = f"""Context:
Question: {query}"""

    try:
        response = chat_session.send_message(prompt) # Send message to the chat session
        return response.text.strip() # Get the text answer and strip whitespace

    except Exception as e: # Catching general exceptions for robustness
        return f"Error getting answer from Gemini API: {e}"


# Main function (Corrected and Enhanced with Gemini Integration and Separate Collections)
def main():
    pdf_path = '/home/admin1/Documents/multimodal/data/book-54-56.pdf' # Make sure your PDF is in the 'data' folder
    data = extract_pdf_data(pdf_path)

    # --- FIRST RUN (INDEXING RUN) ---
    print("--- Starting SCRIPT in INDEXING MODE ---") # ADDED: Script mode indicator
    collections = index_data(data) # Get both text and image collections
    text_collection = collections["text_collection"] # Access text collection
    image_collection = collections["image_collection"] # Access image collection
    print("--- Collections indexed and persisted. ---") # ADDED: Indexing completion confirmation
    chroma_client.persist() # ADDED: Explicit persist call for extra safety

    # --- SUBSEQUENT RUNS (QUERYING RUN) - COMMENT OUT INDEXING ABOVE and UNCOMMENT LOADING BELOW ---
    # print("--- Starting SCRIPT in QUERYING MODE ---") # ADDED: Script mode indicator
    # try:
    #     text_collection = chroma_client.get_collection(name="pdf_text_data")
    #     image_collection = chroma_client.get_collection(name="pdf_image_data")
    #     print("Collections LOADED from disk.") # ADDED: Loading confirmation
    # except Exception as e:
    #     print(f"Error loading collections: {e}")
    #     print("Please run the script ONCE in INDEXING MODE first.")
    #     return
    #
    # print("--- Ready for querying. ---") # ADDED: Query ready confirmation


    query_text = "What Makes Something Trans-disciplinary?" # Example query - you can change this

    text_query_results = query_chromadb(text_collection, query_text) # Query TEXT collection

    retrieved_context_chunks = []
    references = []

    print("\nRetrieved Text Context Chunks from ChromaDB (Text Collection):")
    for i in range(len(text_query_results['ids'][0])):
        page_num = text_query_results['metadatas'][0][i]['page_num']
        result_type = text_query_results['metadatas'][0][i]['type']

        if result_type == 'text':
            chunk_index = text_query_results['metadatas'][0][i]['chunk_index']
            document_text = text_query_results['documents'][0][i] # Access the retrieved text chunk
            retrieved_context_chunks.append(document_text) # Add to context for Gemini
            references.append(f"Page {page_num}, Text Chunk {chunk_index}") # Store reference
            print(f"\n--- Text Result {i+1} (Page {page_num}, Chunk {chunk_index}) ---")
            print(f"Content: {document_text}")


        elif result_type == 'image': # This part should not be reached when querying text_collection, but kept for completeness
            image_index = text_query_results['metadatas'][0][i]['image_index']
            references.append(f"Page {page_num}, Image {image_index}") # Store image reference
            print(f"\n--- Image Result {i+1} (Page {page_num}, Image {image_index}) ---")
            print(f"Reference: Page {page_num}, Image {image_index}") # Image reference - you'd need to handle displaying the image separately


    # Get answer from Gemini using retrieved text context
    gemini_answer = get_gemini_answer(query_text, retrieved_context_chunks)

    print("\n\n--- Gemini Answer ---")
    print(f"Question: {query_text}")
    print(f"Answer: {gemini_answer}")

    print("\n--- References (from Text Collection) ---")
    for ref in references:
        print(ref)


if __name__ == "__main__":
    main()