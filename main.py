import logging 
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf,embed_texts
from vector_db import qdrantStorage
from custom_types import RAGChunksAndSrc,RAGQueryResult,RAGSearchResult,RAGUpsertResult 

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
        chunks=load_and_chunk_pdf(pdf_path)
        return RAGChunksAndSrc(chunks=chunks,source_id=source_id)
    
    def _uppsert(chunks_and_src:RAGChunksAndSrc)->RAGUpsertResult:
        chunks=chunks_and_src.chunks
        source_id=chunks_and_src.source_id
        vectors=embed_texts(chunks)
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,f"{source_id}:{i}"))for i in range(len(chunks))]
        payloads=[{"source":source_id,"text":chunks[i]} for i in range(len(chunks))]
        qdrantStorage().upsert(ids,vectors,payloads)
        return RAGUpsertResult(inngested=len(chunks))


    chunks_and_src=await ctx.step.run("load_and_chunk",lambda:_load(ctx),output_type=RAGChunksAndSrc)
    inngested=await ctx.step.run("embed_and_upsert",lambda:_uppsert(chunks_and_src),output_type=RAGUpsertResult)
    return inngested.model_dump()
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx:inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = qdrantStorage() 
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"],sources=found["sources"]
    )
    question=ctx.event.data["question"]
    top_k=int(ctx.event.data.get("top_k",5))

    found=await ctx.step.run("embed-and-search",lambda: _search(question,top_k),output_type=RAGSearchResult)
    
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"   
        "Answer concisely using the context above."
    )
    adapter = ai.google.Adapter(
    auth_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-2.5-flash"
    )
    res=await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer questions using only the provided context."
                },
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
        }
    )
    answer=res["choices"][0]["messages"]["content"].strip()
    return {"answer":answer,"sources":found.sources,"num_contexts":len(found.contexts)}

inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf,rag_query_pdf_ai])


