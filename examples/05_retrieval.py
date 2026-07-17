"""
CONCEPT 5 — Retrieval is not just vector search.

Run:  python examples/05_retrieval.py   (runs offline — no API key needed)

Takeaway: production RAG is a pipeline, not one call:
    retrieve (wide net)  →  rerank (sharpen)  →  filter (keep the best few)
Reranking + filtering improve answer quality more than most people expect.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support_agent.retrieval import retrieve, rerank, filter_docs

query = "my payment failed, how do I fix it?"
print(f"Query: {query!r}\n")

# Stage 1 — RETRIEVE: cast a wide net (like vector_store.search(query, k=5)).
candidates = retrieve(query, k=5)
print("1) RETRIEVE (top 5 by similarity):")
for d in candidates:
    print(f"   {d['score']:.3f}  {d['title']}")

# Stage 2 — RERANK: re-score with a sharper (query, doc) signal.
reranked = rerank(query, candidates)
print("\n2) RERANK (reordered by relevance):")
for d in reranked:
    print(f"   {d['rerank_score']:.3f}  {d['title']}")

# Stage 3 — FILTER: drop weak matches, keep only what the LLM should read.
final = filter_docs(reranked)
print("\n3) FILTER (what actually gets sent to the LLM):")
for d in final:
    print(f"   ✓ {d['title']}")

print(f"\nWent from {len(candidates)} candidates down to {len(final)} high-signal docs.")
print("A demo stops at step 1. Production does all three.")
