import os
import tempfile
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv()

def configure_llama_index():
    Settings.llm = AzureOpenAI(
        model="gpt-4o",
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_VERSION")
    )
    Settings.embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="text-embedding-ada-002",
        api_key=os.getenv("AZURE_EMBEDDING_KEY"),
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_version="2025-01-01-preview"
    )

def load_pdf_llamaindex(file_path: str) -> list:
    configure_llama_index()
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    chunks = [doc.text for doc in documents if doc.text.strip()]
    print(f"LlamaIndex loaded {len(documents)} document nodes")
    return chunks

def load_pdf_bytes_llamaindex(file_bytes: bytes) -> list:
    configure_llama_index()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    reader = SimpleDirectoryReader(input_files=[tmp_path])
    documents = reader.load_data()
    chunks = [doc.text for doc in documents if doc.text.strip()]
    os.unlink(tmp_path)
    print(f"LlamaIndex loaded {len(documents)} document nodes")
    return chunks