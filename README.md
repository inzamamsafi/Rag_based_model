# Question Answering System

This project is a Question Answering System from pdf book that uses a FastAPI backend and a Streamlit frontend. The system processes questions and provides answers based on the content of a PDF file.

## Prerequisites

- Python 3.10.12
- `pip` (Python package installer)
- `.env` file with the necessary environment variables

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine (branch name staging)

### 2. Create and Activate a Virtual Environment

It is recommended to create a virtual environment to manage dependencies.

### 3. Install Dependencies

Install the required Python packages using `pip`.


### 4. Set Up Environment Variables

Create a `.env` file in the root directory of the project and put the variable name as mentioned below
GEMINI_API_KEY=your_gemini_api_key_here


### 5. Run the FastAPI Backend

Start the FastAPI backend server

Run this command: uvicorn app:app --reload


This will start the FastAPI server at `http://127.0.0.1:8000`.

### 6. Run the Streamlit Frontend

In a new terminal (with the virtual environment activated), start the Streamlit app.
Run this command: streamlit run streamlit_app.py

This will start the Streamlit app in your default web browser.

## Usage

1. Open the Streamlit app in your browser.
2. Enter a question in the input field.
3. Click the "Submit" button.
4. The app will process your question and display the answer.

## Note: 
The embeding of the pdf is persisted. In case if it does not exist then first time when user will ask question then embeding and indexing will happen and it will take some time. But from 2nd time onward it will work with good speed and embeding and indexing won't happen.