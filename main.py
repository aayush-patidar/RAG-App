import logging
import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
import inngest
import inngest.fast_api
from custom_types import (RAGChunksAndSrc,RAGSearchResult,RAGUpsertResult,)
from data_loader import (load_and_chunk_pdf,embed_texts,)
from vector_db import qdrantStorage
load_dotenv()
logger = logging.getLogger("uvicorn")
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

app = FastAPI()
@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunksAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        logger.info(f"Loading PDF: {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path)
        logger.info(f"Extracted {len(chunks)} chunks")
        return RAGChunksAndSrc(
            chunks=chunks,
            source_id=source_id,
        )

    def _upsert(chunks_and_src: RAGChunksAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        if len(chunks) == 0:
            raise ValueError("No chunks extracted from PDF.")
        vectors = embed_texts(chunks)
        if len(vectors) == 0:
            raise ValueError("Embedding generation failed.")

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]

        logger.info(f"Upserting {len(ids)} vectors into Qdrant")

        store = qdrantStorage()
        store.upsert(ids, vectors, payloads)

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunksAndSrc,
    )

    result = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return result.model_dump()

@inngest_client.create_function(
    fn_id="rag-query-pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data.get("question")
    if not question:
        raise ValueError("Missing question.")
    top_k = int(ctx.event.data.get("top_k", 5))
    def _search(question: str, top_k: int) -> RAGSearchResult:
        query_vector = embed_texts([question])[0]
        store = qdrantStorage()
        result = store.search(
            query_vector=query_vector,
            top_k=top_k,
        )
        return RAGSearchResult(
            contexts=result["contexts"],
            sources=result["sources"],
        )
    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    if len(found.contexts) == 0:

        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
            "num_contexts": 0,
        }

    context = "\n\n".join(found.contexts)

    prompt = f"""
You are a RAG assistant.

Answer ONLY from the supplied context.

If the answer is unavailable, reply exactly:

I couldn't find that information in the provided documents.

Context:

{context}

Question:

{question}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    answer = response.text.strip()
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }
inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf,rag_query_pdf_ai,],)