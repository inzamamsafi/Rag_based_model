import streamlit as st
import requests
import time

# FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8000/ask"

# Function to call the FastAPI backend
def ask_question(question):
    try:
        response = requests.post(FASTAPI_URL, json={"question": question})
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit App UI
def main():
    st.title("Question Answering System")
    st.markdown("Ask a question and get an answer based on the content of a PDF.")

    # Initialize session state for response and progress
    if "response" not in st.session_state:
        st.session_state.response = None
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Input field for the question
    question = st.text_input("Enter your question:")

    # Submit button
    if st.button("Submit"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Clear previous response
            st.session_state.response = None
            st.session_state.processing = True

            # Show progress icon (yellow)
            with st.empty():
                st.markdown("⏳ Processing...")
                time.sleep(1)  # Simulate processing delay

            # Call the FastAPI backend
            st.session_state.response = ask_question(question)

            # Update progress icon (green)
            st.session_state.processing = False

    # Display progress icon
    if st.session_state.processing:
        st.markdown("⏳ Processing...", help="The app is processing your question.")
    elif st.session_state.response is not None:
        st.markdown("✅ Done!", help="Processing complete.")

    # Display the response
    if st.session_state.response is not None:
        st.subheader("Answer:")
        st.write(st.session_state.response)

# Run the Streamlit app
if __name__ == "__main__":
    main()


# CODE RUN

# uvicorn app:app --reload