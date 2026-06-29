import logging 
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_end_chunk_pdf,embed_texts
from vector_db import qdrantStorage
from custom_types import RAGChunksAndSrc,RAGQueryResult,RAGSearchResulr,RAGUpsertResult 

load_dotenv()

inngest_client=inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

app=FastAPI()
@inngest_client.create_function(
    fn_id="rag",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")

)
async def rag_ingest_pdf(ctx:inngest.Context):
    def _load(ctx :inngest.Context)-> RAGChunksAndSrc:
        pdf_path=ctx.event.data["pdf_path"]
        source_id=ctx.event.data.get("source_id",pdf_path)
        chunks=load_end_chunk_pdf(pdf_path)
        return RAGChunksAndSrc(chunks=chunks,source_id=source_id)
    
    def _uppsert(chunks_and_src:RAGChunksAndSrc)->RAGUpsertResult:
        chunks=chunks_and_src.chunks
        source_id=chunks_and_src.source_id
        vectors=embed_texts[chunks]
        ids=[str(uuid.uuids(uuid.NAMESPACE_URL),name=f"{source_id}:{i}")for i in range(len(chunks))]
        payloads=[{"source":source_id,"text":chunks[i]} for i in range(len(chunks))]
        qdrantStorage().upsert(ids,vectors,payloads)
        return RAGUpsertResult(inngested=len(chunks))


    chunks_and_src=await ctx.step.run("load_and_chunk",lambda:_load(ctx),output_type=RAGChunksAndSrc)
    inngested=await ctx.step.run("embed_and_upsert",lambda:_uppsert(chunks_and_src),output_type=RAGUpsertResult)
    return inngested.model_dump()

    
inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf])


