import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from rag.document_store import get_vectorstore

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def qa_agent(question: str) -> dict:
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        return {
            "answer": "No relevant content found in the document.",
            "sources": [],
            "context": ""
        }

    context = "\n\n".join([doc.page_content for doc in results])
    sources = [doc.page_content[:100] + "..." for doc in results]

    prompt = "Answer this question using only the context below.\n\nCONTEXT:\n" + context + "\n\nQUESTION:\n" + question + "\n\nANSWER:"

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful document assistant. Answer questions based only on the provided context. If the answer is not in the context say so clearly."},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "context": context
    }