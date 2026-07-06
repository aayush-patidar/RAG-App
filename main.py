import logging
import os
import uuid
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from custom_types import (
    RAGChunksAndSrc,
    RAGSearchResult,
    RAGUpsertResult,
)
from data_loader import (
    load_and_chunk_pdf,
    embed_texts,
)
from vector_db import qdrantStorage


load_dotenv()

logger = logging.getLogger("uvicorn")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

inngest_client = inngest.Inngest(
    app_id="rag_appa",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

app = FastAPI()


@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(
        event="rag/ingest_pdf"
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load() -> RAGChunksAndSrc:
        pdf_path = ctx.event.data["pdf_path"]

        user_id = ctx.event.data["user_id"]

        document_id = ctx.event.data.get(
            "document_id",
            str(uuid.uuid4()),
        )

        source_id = ctx.event.data.get(
            "source_id",
            os.path.basename(pdf_path),
        )

        chunks = load_and_chunk_pdf(pdf_path)

        if not chunks:
            raise ValueError(
                "No text chunks extracted from PDF."
            )

        return RAGChunksAndSrc(
            chunks=chunks,
            source_id=source_id,
            user_id=user_id,
            document_id=document_id,
        )

    def _upsert(
        chunks_and_src: RAGChunksAndSrc,
    ) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        user_id = chunks_and_src.user_id
        document_id = chunks_and_src.document_id

        vectors = embed_texts(chunks)

        if not vectors:
            raise ValueError(
                "Embedding generation failed."
            )

        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{user_id}:{document_id}:{i}",
                )
            )
            for i in range(len(chunks))
        ]

        payloads = [
            {
                "user_id": user_id,
                "document_id": document_id,
                "source": source_id,
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]

        store = qdrantStorage()

        store.upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        _load,
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
    trigger=inngest.TriggerEvent(
        event="rag/query_pdf_ai"
    ),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data.get("question")
    user_id = ctx.event.data.get("user_id")

    if not question:
        raise ValueError("Missing question.")

    if not user_id:
        raise ValueError("Missing user_id.")

    top_k = int(
        ctx.event.data.get("top_k", 5)
    )

    def _search() -> RAGSearchResult:
        query_vector = embed_texts(
            [question]
        )[0]

        store = qdrantStorage()

        result = store.search(
            query_vector=query_vector,
            user_id=user_id,
            top_k=top_k,
        )

        return RAGSearchResult(
            contexts=result["contexts"],
            sources=result["sources"],
        )

    found = await ctx.step.run(
        "embed-and-search",
        _search,
        output_type=RAGSearchResult,
    )

    if not found.contexts:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the uploaded documents."
            ),
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

    return {
        "answer": response.text.strip(),
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


inngest.fast_api.serve(
    app,
    inngest_client,
    [
        rag_ingest_pdf,
        rag_query_pdf_ai,
    ],
)