"""
CONCEPT 1 — State  (and CONCEPT 8 — Memory)

State is the shared memory that flows through the whole workflow. Every node
reads from it and returns a partial update that LangGraph merges back in.

Think of it as a single source of truth: instead of passing ad-hoc arguments
between steps, each step perceives everything that happened before it. That is
what makes multi-step agents debuggable and predictable.

Concept 8 (Memory) lives here too: fields like `retrieved_articles`,
`classification`, and `history` are *operational memory* — the agent remembers
its own decisions and tool outputs, not just the chat transcript.
"""

from operator import add
from typing import Annotated, TypedDict


class SupportState(TypedDict, total=False):
    """
    Shared state for the customer-support agent.

    `total=False` means every key is optional — nodes fill fields in as the
    workflow progresses, starting from just the incoming ticket.
    """

    # ── Input ────────────────────────────────────────────────────
    ticket_id: str
    customer_message: str
    customer_tier: str  # e.g. "Free" | "Pro" | "Enterprise"

    # ── Retrieval (concept 5) ────────────────────────────────────
    retrieved_articles: list  # knowledge-base snippets used to answer

    # ── Routing + classification (concepts 4 & 6) ────────────────
    route: str          # which specialist handled it
    classification: dict  # validated structured output, stored as a JSON-safe dict

    # ── Generation ───────────────────────────────────────────────
    draft_response: str
    final_response: str

    # ── Human-in-the-loop (concept 10) ───────────────────────────
    escalation_required: bool
    human_decision: str  # "approve" | "edit" | "reject" set by a reviewer

    # ── Memory (concept 8) ───────────────────────────────────────
    # `Annotated[..., add]` tells LangGraph to *append* to this list on each
    # update instead of overwriting it — an audit trail of everything the
    # agent did, which is exactly what operational memory looks like.
    history: Annotated[list, add]


def new_ticket_state(
    ticket_id: str,
    customer_message: str,
    customer_tier: str = "Free",
) -> SupportState:
    """Convenience constructor for a fresh ticket entering the graph."""
    return SupportState(
        ticket_id=ticket_id,
        customer_message=customer_message,
        customer_tier=customer_tier,
        retrieved_articles=[],
        history=[],
    )
