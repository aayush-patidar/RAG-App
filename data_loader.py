from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
load_dotenv()
client=OpenAI()
EMBED_MODEL="test-embedding-3-large"
EMBED_DIM=3072

splitter=SentenceSplitter(chunk_size=1000,chunk_overlap=200)
def laod_end_chunk_pdf(path:str):
    docs=PDFReader().load_data(files=path)
    texts=[d.text for d in docs if getattr(d,"text",None)]
    chunks=[]
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts:list[str]-> list[list[float]]):
    response=client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]