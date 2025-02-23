import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import faiss
import os
import pickle
from transformers import CLIPProcessor, CLIPModel

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyANVmZMHBPVIyEmX29g1qHuJhRdbQT_cUE"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Create the Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21", # gemini-1.5-flash # gemini-2.0-flash-thinking-exp-01-21
    generation_config=generation_config,
)

# Load pre-trained CLIP model and processor
clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)

# Function to convert PDF to images and save them
def pdf_to_images_and_save(pdf_path, save_dir="images"):
    """Converts a PDF to images and saves each image with the page number as the filename."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = convert_from_path(pdf_path)
    image_paths = []  # Store paths to loaded images

    for i, image in enumerate(images):
        image_path = os.path.join(save_dir, f"page_{i + 1}.png")  # Page numbers start from 1
        image.save(image_path)
        image_paths.append(image_path)

    return image_paths

# Function to load images from saved paths
def load_images_from_paths(image_paths):
    """Loads images from a list of file paths."""
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            images.append(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_path}")
            # Handle missing images (e.g., skip, use a placeholder)
            # For simplicity, we'll just skip them here.
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
def save_embeddings_and_index(embeddings, faiss_index, image_paths, save_dir="data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save embeddings
    with open(os.path.join(save_dir, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    # Save FAISS index
    faiss.write_index(faiss_index, os.path.join(save_dir, "faiss_index.index"))

    # Save image paths
    with open(os.path.join(save_dir, "image_paths.pkl"), "wb") as f:
        pickle.dump(image_paths, f)

# Function to load embeddings and FAISS index from disk
def load_embeddings_and_index(save_dir="data"):
    embeddings_path = os.path.join(save_dir, "embeddings.pkl")
    faiss_index_path = os.path.join(save_dir, "faiss_index.index")
    image_paths_path = os.path.join(save_dir, "image_paths.pkl")

    if (not os.path.exists(embeddings_path) or
        not os.path.exists(faiss_index_path) or
        not os.path.exists(image_paths_path)):
        return None, None, None

    # Load embeddings
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)

    # Load FAISS index
    faiss_index = faiss.read_index(faiss_index_path)

    # Load image paths
    with open(image_paths_path, "rb") as f:
        image_paths = pickle.load(f)

    return embeddings, faiss_index, image_paths

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

# Modified query_gemini_with_images function (with system instruction)
def query_gemini_with_images(images, question):
    # --- System Instruction ---
    system_instruction = (
        "You are a helpful assistant that answers questions based on the provided images of a document. "
        "Be concise, accurate, and reference the relevant page numbers if possible.  "
        "If the answer cannot be found in the images, respond with 'I cannot find the answer in the provided images.'"
    )

    # Combine system instruction, question, and images for Gemini
    inputs = [system_instruction, question] + images  # System instruction comes FIRST

    response = model.generate_content(inputs)
    return response.text


# Main workflow
if __name__ == "__main__":
    # Step 1: Check if embeddings and FAISS index already exist
    save_dir = "data"
    embeddings, faiss_index, image_paths = load_embeddings_and_index(save_dir)

    if embeddings is None or faiss_index is None or image_paths is None:
        # Step 2: Convert PDF to images and save them (only if embeddings/index don't exist)
        pdf_path = "data/book.pdf"  # Replace with your PDF file path
        image_paths = pdf_to_images_and_save(pdf_path) # Save and get paths
        images = load_images_from_paths(image_paths) # Load the images

        # Step 3: Generate embeddings and store in FAISS (only if embeddings/index don't exist)
        faiss_index, embeddings = store_embeddings_in_faiss(images)

        # Save embeddings, FAISS index, *and image paths* to disk
        save_embeddings_and_index(embeddings, faiss_index, image_paths, save_dir)
    else:
        # Load images from saved paths
        images = load_images_from_paths(image_paths)

    # Step 4: Ask a question and retrieve top-K relevant images
    question = "What is a premise?? provide reference as well. provide reference as well"
    top_k_images = retrieve_top_k_images(question, faiss_index, images, k=3)

    # Step 5: Generate an answer using Gemini
    answer = query_gemini_with_images(top_k_images, question)

    print("Answer:", answer)