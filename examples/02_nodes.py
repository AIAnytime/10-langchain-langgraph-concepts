"""
CONCEPT 2 — A node is just a function.

Run:  python examples/02_nodes.py   (runs offline — no API key needed)

Takeaway: there's no magic. A node is a plain function `state -> partial update`.
Each does ONE thing. Power comes from orchestration, not node complexity.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support_agent.retrieval import retrieve, rerank, filter_docs

# Three real nodes from our agent's retrieval stage — each is a small function.
def retrieve_node(state):
    return {"candidates": retrieve(state["user_query"], k=5)}

def rerank_node(state):
    return {"candidates": rerank(state["user_query"], state["candidates"])}

def filter_node(state):
    return {"final_docs": filter_docs(state["candidates"])}

# Run them one after another, threading the shared state through.
state = {"user_query": "why did my payment fail?"}
for node in (retrieve_node, rerank_node, filter_node):
    state.update(node(state))
    print(f"{node.__name__:14} -> keys now: {list(state.keys())}")

print("\nFinal documents this node-chain selected:")
for d in state["final_docs"]:
    print(f"  - {d['title']}  (score={d['rerank_score']:.3f})")

print("\nEach node is trivial on its own. Stringing them together is the point.")
