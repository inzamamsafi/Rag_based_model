import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import faiss
import os
from transformers import CLIPProcessor, CLIPModel

# Configure Gemini API with the API key directly in the code
GEMINI_API_KEY = "AIzaSyANVmZMHBPVIyEmX29g1qHuJhRdbQT_cUE"  # Replace with your actual Gemini API key
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
    model_name="gemini-2.0-pro-exp-02-05",  # Use the latest Gemini model
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

    return index

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

# Main workflow
if __name__ == "__main__":
    # Step 1: Convert PDF to images
    pdf_path = "data/book.pdf"  # Replace with your PDF file path
    images = pdf_to_images(pdf_path)

    # Step 2: Generate embeddings and store in FAISS
    faiss_index = store_embeddings_in_faiss(images)

    # Step 3: Ask a question and retrieve top-K relevant images
    question = "What Makes Something Trans-disciplinary"
    top_k_images = retrieve_top_k_images(question, faiss_index, images, k=3)

    # Step 4: Generate an answer using Gemini
    answer = query_gemini_with_images(top_k_images, question)

    print("Answer:", answer)