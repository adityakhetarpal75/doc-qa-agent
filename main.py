from graph.workflow import build_graph

graph = build_graph()

questions = [
    "What is the main topic of this document?",
    "What are the key findings?",
    "What recommendations are made?"
]

for question in questions:
    print(f"\nQ: {question}")
    result = graph.invoke({
        "question": question,
        "answer": "",
        "sources": [],
        "context": ""
    })
    print(f"A: {result['answer']}")