import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(
        file=Path(path)
    )

    texts = [
        doc.text.strip()
        for doc in docs
        if getattr(doc, "text", None) and doc.text.strip()
    ]

    chunks = []

    for text in texts:
        chunks.extend(
            splitter.split_text(text)
        )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings = []

    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )

        embeddings.append(
            response.embeddings[0].values
        )

    return embeddings