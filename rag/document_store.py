import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# SWITCHED FROM AZURE TO OLLAMA - NO RATE LIMITS
embeddings = OllamaEmbeddings(model="nomic-embed-text")

INDEX_NAME = "doc-agent1"

def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,        # nomic-embed-text = 768 dims (NOT 1536)
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

def store_documents(chunks: list[str]) -> PineconeVectorStore:
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=INDEX_NAME
    )
    return vectorstore

def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )