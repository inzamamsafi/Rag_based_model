from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import faiss
import os
import pickle
from transformers import CLIPProcessor, CLIPModel

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Read from .env file
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# Create the Gemini model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-pro-exp-02-05",  # gemini-1.5-flash # gemini-2.0-flash-thinking-exp-01-21
    generation_config=generation_config,
)

# Load pre-trained CLIP model and processor
clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)

# Function to convert PDF to images
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

# Function to generate question embedding using CLIP
def generate_question_embedding(question):
    inputs = clip_processor(text=[question], return_tensors="pt", padding=True, truncation=True)
    outputs = clip_model.get_text_features(**inputs)
    embedding = outputs.detach().cpu().numpy().astype('float32').flatten()
    return embedding

# Function to generate image embedding using CLIP
def generate_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    embedding = outputs.detach().cpu().numpy().astype('float32').flatten()
    return embedding

# Function to store embeddings and FAISS index to disk
def save_embeddings_and_index(embeddings, faiss_index, save_dir="embedding"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save embeddings
    with open(os.path.join(save_dir, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    
    # Save FAISS index
    faiss.write_index(faiss_index, os.path.join(save_dir, "faiss_index.index"))

# Function to load embeddings and FAISS index from disk
def load_embeddings_and_index(save_dir="embedding"):
    embeddings_path = os.path.join(save_dir, "embeddings.pkl")
    faiss_index_path = os.path.join(save_dir, "faiss_index.index")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(faiss_index_path):
        return None, None
    
    # Load embeddings
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    
    # Load FAISS index
    faiss_index = faiss.read_index(faiss_index_path)
    
    return embeddings, faiss_index

# Function to store embeddings in FAISS
def store_embeddings_in_faiss(images):
    dimension = 512  # CLIP ViT-base-patch32 embedding dimension
    index = faiss.IndexFlatL2(dimension)

    embeddings = []
    for image in images:
        embedding = generate_image_embedding(image)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    index.add(embeddings)

    return index, embeddings

# Function to retrieve top-K relevant images
def retrieve_top_k_images(question, faiss_index, images, k=3):
    # Generate embedding for the question
    question_embedding = generate_question_embedding(question)

    # Search for top-K similar images in the FAISS index
    distances, indices = faiss_index.search(np.array([question_embedding]), k)

    # Retrieve the top-K images
    top_k_images = [images[i] for i in indices[0]]
    return top_k_images

# Function to query Gemini with images and a question
def query_gemini_with_images(images, question):
    # Combine images and question for Gemini
    inputs = [question] + images
    response = model.generate_content(inputs)
    return response.text

# FastAPI Application
app = FastAPI()

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Step 1: Check if embeddings and FAISS index already exist
        save_dir = "embedding"
        embeddings, faiss_index = load_embeddings_and_index(save_dir)
        
        if embeddings is None or faiss_index is None:
            # Step 2: Convert PDF to images (only if embeddings/index don't exist)
            pdf_path = "data/book.pdf"  # Replace with your PDF file path
            images = pdf_to_images(pdf_path)
            
            # Step 3: Generate embeddings and store in FAISS (only if embeddings/index don't exist)
            faiss_index, embeddings = store_embeddings_in_faiss(images)
            
            # Save embeddings and FAISS index to disk
            save_embeddings_and_index(embeddings, faiss_index, save_dir)
        else:
            # Load images from disk (if needed)
            pdf_path = "data/book.pdf"
            images = pdf_to_images(pdf_path)
        
        # Step 4: Retrieve top-K relevant images
        question = request.question
        top_k_images = retrieve_top_k_images(question, faiss_index, images, k=4)
        
        # Step 5: Generate an answer using Gemini
        answer = query_gemini_with_images(top_k_images, question)
        
        return {"answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)