import streamlit as st
from graph.workflow import build_graph
from rag.document_store import store_documents
from rag.llama_loader import load_pdf_bytes_llamaindex as load_pdf_bytes
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ADD THIS

st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📄",
    layout="wide"
)
st.title("📄 Document Q&A Agent")
st.caption("Powered by CodeLlama + Pinecone RAG")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with st.spinner("Reading and storing document in Pinecone..."):
        pages = load_pdf_bytes(uploaded_file.read())

        # SPLIT PAGES INTO SMALLER CHUNKS FOR NOMIC-EMBED-TEXT
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = []
        for page in pages:
            chunks.extend(splitter.split_text(page))

        store_documents(chunks)

    st.success(f"Document loaded — {len(chunks)} chunks stored in Pinecone")

    question = st.text_input("Ask a question about the document", placeholder="What are the key findings?")

    if st.button("🔍 Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and answering..."):
                graph = build_graph()
                result = graph.invoke({
                    "question": question,
                    "answer": "",
                    "sources": [],
                    "context": ""
                })
            st.subheader("💬 Answer")
            st.markdown(result["answer"])
            with st.expander("📚 Retrieved Sources"):
                for i, source in enumerate(result["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(source)
            with st.expander("🔍 Full Context Used"):
                st.text(result["context"])
            st.info("Traces logged to LangSmith")