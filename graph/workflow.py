from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.qa_agent import qa_agent
from rag.document_store import store_documents
from rag.pdf_loader import load_pdf_bytes

class DocAgentState(TypedDict):
    question: str
    answer: str
    sources: list
    context: str

def run_qa(state: DocAgentState) -> DocAgentState:
    print("Running QA Agent...")
    result = qa_agent(state["question"])
    state["answer"] = result["answer"]
    state["sources"] = result["sources"]
    state["context"] = result.get("context", "")
    return state

def build_graph():
    graph = StateGraph(DocAgentState)
    graph.add_node("qa", run_qa)
    graph.set_entry_point("qa")
    graph.add_edge("qa", END)
    return graph.compile()