from pydantic import BaseModel
from typing import Optional


class RAGChunksAndSrc(BaseModel):
    chunks: list[str]
    source_id: Optional[str] = None
    user_id: str
    document_id: str


class RAGUpsertResult(BaseModel):
    ingested: int


class RAGSearchResult(BaseModel):
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int