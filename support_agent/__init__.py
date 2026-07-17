"""
SupportGraph — a production-style customer-support agent built with
LangChain + LangGraph that demonstrates the 10 concepts every AI
engineer should know.

Package layout (each file maps to one or more concepts):

    config.py          → OpenRouter LLM wiring
    state.py           → 1. State, 8. Memory (shared, typed state)
    schemas.py         → 6. Structured outputs (Pydantic)
    knowledge_base.py  → data for retrieval
    retrieval.py       → 5. Retrieval (retrieve → rerank → filter)
    nodes.py           → 2. Nodes, 4. Routing, 10. Human-in-the-loop
    graph.py           → 3. Graph, 9. Checkpointing (assembly)
"""

__all__ = ["config", "state", "schemas", "knowledge_base", "retrieval", "nodes", "graph"]
