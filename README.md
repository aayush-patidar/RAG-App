# PDF RAG Application

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content.

The application extracts text from PDFs, splits the text into chunks, converts chunks into embeddings, stores them in Qdrant, retrieves relevant content, and uses Google Gemini to generate answers.

The backend is built using FastAPI and the frontend is built using Streamlit.

---

## Features

- Upload PDF files
- Extract text from PDF documents
- Split PDF text into smaller chunks
- Generate embeddings for document chunks
- Store embeddings in Qdrant vector database
- Search relevant chunks using semantic search
- Ask questions about uploaded PDFs
- Generate answers using Google Gemini
- FastAPI backend API
- Streamlit frontend interface

---

## Tech Stack

- Python
- FastAPI
- Streamlit
- Google Gemini API
- Qdrant Vector Database
- PDF text extraction
- Embeddings
- Uvicorn
- uv package manager

---

## How It Works

1. User uploads a PDF file using the Streamlit frontend.
2. Streamlit sends the PDF to the FastAPI backend.
3. FastAPI reads and extracts text from the PDF.
4. The extracted text is split into smaller chunks.
5. Each chunk is converted into an embedding.
6. Embeddings are stored in Qdrant.
7. User asks a question about the uploaded PDF.
8. The application searches Qdrant for relevant chunks.
9. Relevant chunks and the user question are sent to Google Gemini.
10. Gemini generates an answer based on the PDF content.
11. The answer is displayed in the Streamlit frontend.

---

## Project Structure

```text
PDF-RAG-Application/
│
├── uploads/                # Stores uploaded PDF files
├── .gitignore              # Ignored files and folders
├── .python-version         # Python version configuration
├── README.md               # Project documentation
├── custom_types.py         # Custom types and models
├── data_loader.py          # PDF loading and text extraction
├── main.py                 # FastAPI backend application
├── streamlit_app.py        # Streamlit frontend application
├── vector_db.py            # Qdrant vector database operations
├── pyproject.toml          # Project dependencies and configuration
└── uv.lock                 # Locked dependency versions
