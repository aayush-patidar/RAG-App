# PDF RAG Application

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF files and ask questions based only on the uploaded document content.

The project uses Gemini embeddings and Gemini Flash Lite for answers, Qdrant as the vector database, FastAPI and Inngest for backend workflows, and Streamlit for the frontend.

## Features

- Upload PDF files from a Streamlit interface
- Extract text from PDFs
- Split PDF text into chunks
- Generate embeddings using Gemini
- Store embeddings in Qdrant
- Search relevant PDF chunks using semantic search
- Generate answers using Gemini
- Show document sources used for the answer
- Separate document data by `user_id` so users do not access each other's PDFs
- Background ingestion and query workflows using Inngest
- Docker support for Qdrant

## Tech Stack

- Python
- FastAPI
- Streamlit
- Inngest
- Gemini API
- Qdrant
- Docker
- LlamaIndex PDFReader
- UV package manager

## Project Structure

```text
RAG/
│
├── custom_types.py          # Pydantic models used in workflows
├── data_loader.py           # PDF loading, chunking, and Gemini embeddings
├── vector_db.py             # Qdrant storage, upsert, and filtered search
├── main.py                  # FastAPI and Inngest functions
├── streamlit_app.py         # Streamlit frontend
├── docker-compose.yml       # Qdrant Docker configuration
├── pyproject.toml           # Python dependencies
├── uv.lock                  # Locked dependency versions
├── .python-version          # Python version
├── .env.example             # Environment variable template
├── .gitignore               # Files ignored by Git
└── README.md                # Project documentation
```

The following files and folders are ignored by Git:

```text
.env
.venv/
__pycache__/
qdrant_storage/
uploads/
```

## Prerequisites

Install these before starting:

- Python 3.11 or newer
- Docker Desktop
- Node.js and npm
- UV package manager
- Gemini API key

## Gemini API Key Setup

Create a Gemini API key from Google AI Studio.

:contentReference[oaicite:0]{index=0}

Create a `.env` file in the root folder:

```env
GEMINI_API_KEY=your_gemini_api_key_here
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

Do not upload `.env` to GitHub.

## Install Dependencies

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Create and activate the virtual environment:

```bash
uv venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
uv sync
```

## Start Qdrant Using Docker

Run Qdrant:

```bash
docker compose up -d
```

Check that Qdrant is running:

```bash
docker ps
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

To stop Qdrant:

```bash
docker compose down
```

## Start the FastAPI Server

Open a terminal, activate the virtual environment, and run:

```bash
uvicorn main:app --reload --port 8000
```

FastAPI runs at:

```text
http://127.0.0.1:8000
```

Inngest endpoint:

```text
http://127.0.0.1:8000/api/inngest
```

## Start the Inngest Development Server

Open another terminal and run:

```bash
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

Inngest dashboard:

```text
http://127.0.0.1:8288
```

## Start the Streamlit Frontend

Open a third terminal, activate the virtual environment, and run:

```bash
streamlit run streamlit_app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

## How the Application Works

1. A user uploads a PDF from Streamlit.
2. Streamlit saves the PDF temporarily in the `uploads` folder.
3. Streamlit sends a `rag/ingest_pdf` event to Inngest.
4. Inngest triggers the PDF ingestion workflow.
5. The backend extracts text from the PDF.
6. The text is split into smaller chunks.
7. Gemini generates embeddings for each chunk.
8. The chunks and embeddings are stored in Qdrant.
9. When a user asks a question, Gemini creates an embedding for that question.
10. Qdrant finds the most relevant chunks.
11. Gemini generates an answer using only the retrieved chunks.
12. The answer and document sources are shown in Streamlit.

## Inngest Functions

The project contains two Inngest functions.

### PDF Ingestion

Event name:

```text
rag/ingest_pdf
```

This function:

- Loads the PDF
- Extracts text
- Creates chunks
- Generates embeddings
- Stores vectors in Qdrant

### PDF Question Answering

Event name:

```text
rag/query_pdf_ai
```

This function:

- Embeds the user question
- Searches Qdrant
- Retrieves relevant chunks
- Sends the chunks to Gemini
- Returns an answer and sources

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Gemini API key used for embeddings and answer generation |
| `INNGEST_API_BASE` | Local Inngest API URL used by Streamlit |

## Common Problems

### Qdrant connection refused

Error:

```text
WinError 10061
```

Start Qdrant:

```bash
docker compose up -d
```

### Inngest cannot connect to FastAPI

Make sure FastAPI is running first:

```bash
uvicorn main:app --reload --port 8000
```

Then start Inngest:

```bash
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

### No chunks extracted from PDF

This usually happens when the PDF is scanned or image-based and has no selectable text.

Use a text-based PDF or add OCR support later.

### Gemini API key error

Check that `.env` exists and contains:

```env
GEMINI_API_KEY=your_key_here
```

Restart FastAPI after changing `.env`.

## Future Improvements

- Add user authentication
- Add OCR support for scanned PDFs
- Add document deletion
- Add chat history
- Add multiple PDF selection
- Deploy FastAPI and Streamlit
- Use Qdrant Cloud for production
- Add file size validation
- Add rate limiting
- Add better error handling
- Add document preview

## Security Notes

- Never commit `.env`
- Never expose Gemini API keys in frontend code
- Keep Qdrant filtered by `user_id`
- Use authentication before deploying publicly
- Use a separate Qdrant collection or tenant strategy for production users

## License

This project is created for learning and portfolio purposes.
