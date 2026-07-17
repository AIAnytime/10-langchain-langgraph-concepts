"""
CONCEPT 3 — Chains vs Graphs
CONCEPT 9 — Checkpointing

We assemble the small node functions into a directed graph. A linear *chain*
(prompt → LLM → parser) can't express "classify, then branch to a specialist,
then maybe pause for a human". A *graph* expresses that naturally.

    classify ─▶ retrieve ─▶ [route] ─▶ billing / technical / sales / general
                                                     │
                                            [needs_human?]
                                              ┌──────┴───────┐
                                        human_review      finalize
                                              └──────┬───────┘
                                                    END

CONCEPT 9 (Checkpointing): compiling with a checkpointer makes every step
durable. The graph can pause at `human_review` for hours/days and resume from
the exact same spot — nothing is recomputed, nothing is lost.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from support_agent import nodes
from support_agent.state import SupportState


def build_graph(checkpointer=None):
    """
    Wire the nodes and edges into a compiled LangGraph app.

    Pass a checkpointer (e.g. MemorySaver, or a SQLite/Postgres saver in prod)
    to enable durable, resumable execution and human-in-the-loop pauses.
    """
    graph = StateGraph(SupportState)

    # 1. Register nodes (each is just a function from nodes.py).
    graph.add_node("classify", nodes.classify_ticket)
    graph.add_node("retrieve", nodes.retrieve_context)
    graph.add_node("billing", nodes.billing_specialist)
    graph.add_node("technical", nodes.technical_specialist)
    graph.add_node("sales", nodes.sales_specialist)
    graph.add_node("general", nodes.general_specialist)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("finalize", nodes.finalize)

    # 2. Linear part: classify → retrieve.
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")

    # 3. CONCEPT 4 — conditional routing to the right specialist.
    graph.add_conditional_edges(
        "retrieve",
        nodes.route_ticket,
        {"billing": "billing", "technical": "technical",
         "sales": "sales", "general": "general"},
    )

    # 4. CONCEPT 10 — after any specialist, branch to human review or auto-send.
    for specialist in ("billing", "technical", "sales", "general"):
        graph.add_conditional_edges(
            specialist,
            nodes.needs_human,
            {"human_review": "human_review", "finalize": "finalize"},
        )

    # 5. Both terminal paths end the graph.
    graph.add_edge("human_review", END)
    graph.add_edge("finalize", END)

    # CONCEPT 9 — compile with a checkpointer for durable, resumable runs.
    return graph.compile(checkpointer=checkpointer or MemorySaver())
