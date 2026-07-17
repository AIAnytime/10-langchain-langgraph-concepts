"""
CONCEPT 5 — Retrieval is not just vector search

A demo RAG is "vector search → LLM". A production RAG is a *pipeline*:

    retrieve  →  rerank  →  filter  →  (then hand the best few to the LLM)

A vector store might return 20 loosely-relevant docs; only ~5 actually contain
the answer. Reranking and filtering are what turn "relevant-ish" into "correct",
and they matter more for answer quality than most people expect.

To keep this tutorial dependency-free and offline, we implement the *same three
stages* with a small, readable TF-IDF cosine retriever instead of a hosted
vector DB. Swap `retrieve()` for a real vector store and the rest is unchanged.
"""

import math
import re
from collections import Counter

from support_agent.knowledge_base import KNOWLEDGE_BASE

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ── Build a tiny TF-IDF index over the knowledge base (stands in for embeddings)
def _build_index(docs: list[dict]) -> dict:
    doc_tokens = [_tokenize(d["title"] + " " + d["text"]) for d in docs]
    n = len(docs)
    # Inverse document frequency: rare words are more informative.
    df: Counter = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1
    idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}
    return {"doc_tokens": doc_tokens, "idf": idf}


_INDEX = _build_index(KNOWLEDGE_BASE)


def _vectorize(tokens: list[str], idf: dict) -> dict:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    # TF-IDF weight per term; unknown query terms default to a small idf.
    return {t: (c / total) * idf.get(t, 1.0) for t, c in counts.items()}


def _cosine(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ── Stage 1: RETRIEVE — cast a wide net (like `vector_store.search(query, k=20)`)
def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return the top-k candidate documents by TF-IDF cosine similarity."""
    qvec = _vectorize(_tokenize(query), _INDEX["idf"])
    scored = []
    for doc, tokens in zip(KNOWLEDGE_BASE, _INDEX["doc_tokens"]):
        dvec = _vectorize(tokens, _INDEX["idf"])
        scored.append({**doc, "score": _cosine(qvec, dvec)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]


# ── Stage 2: RERANK — re-score candidates with a sharper signal
def rerank(query: str, docs: list[dict]) -> list[dict]:
    """
    Reorder candidates using exact keyword overlap as a cheap cross-encoder
    stand-in. A real system would call a reranker model (e.g. Cohere Rerank,
    bge-reranker) that reads the (query, doc) pair together.
    """
    q_terms = set(_tokenize(query))
    for doc in docs:
        overlap = len(q_terms & set(_tokenize(doc["title"] + " " + doc["text"])))
        # Blend the original retrieval score with direct query-term overlap.
        doc["rerank_score"] = doc.get("score", 0.0) + 0.15 * overlap
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)


# ── Stage 3: FILTER — drop weak matches and keep only the best few
def filter_docs(docs: list[dict], top_n: int = 3, min_score: float = 0.05) -> list[dict]:
    """Keep at most `top_n` docs that clear a relevance floor."""
    strong = [d for d in docs if d.get("rerank_score", 0.0) >= min_score]
    return strong[:top_n]


def retrieve_rerank_filter(query: str) -> list[dict]:
    """The full production-style pipeline in one call."""
    candidates = retrieve(query, k=5)
    reranked = rerank(query, candidates)
    return filter_docs(reranked)
