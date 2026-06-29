from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams,Distance,PointStruct  

class qdrantStorage:
    def __init__(self,url="http://localhost:6333",collections="docs",dim=3072):
        self.client=QdrantClient(url=url,timeout=30)
        self.collection=collections
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim,distance=Distance.COSINE),
            )
        
    def upsert(self,ids,vectors,payloads):
        points=[PointStruct(id=ids[i],vector=vectors[i],payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection,points=points)
    
    #  top k just mean we are looking for this many result from the vector db here we take 5
    def search(self,quert_vector,top_k:int=5):
        results=self.client.search(
            collection_name=self.collection,
            query_vector=quert_vector,
            with_payload=True,
            limit=top_k,
        )
        contexts=[]
        sources=set()
        for r in results:
            payload=getattr(r,"payload",None)or{}
            text=payload.get("text","")
            source=payload.get("source","")
            if text:
                contexts.append(text)
                sources.add(source)
        return {"contexts":contexts,"sources":list(sources)}