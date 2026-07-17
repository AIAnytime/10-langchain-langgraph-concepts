"""
CONCEPT 2 — A node is just a function
CONCEPT 4 — Routing beats one giant prompt
CONCEPT 10 — Human-in-the-loop

Every node below is an ordinary Python function: it takes the shared `state`,
does ONE thing, and returns a partial dict that LangGraph merges back into the
state. No magic. The power comes from how we *orchestrate* these small pieces
in graph.py — not from any single node being clever.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from support_agent.config import get_llm
from support_agent.retrieval import retrieve_rerank_filter
from support_agent.schemas import TicketClassification
from support_agent.state import SupportState

# ── CONCEPT 4: Routing ────────────────────────────────────────────
# Instead of one "you are a billing + technical + sales + compliance expert..."
# mega-prompt, we keep four short, focused personas. Shorter prompts mean higher
# accuracy, lower token cost, and predictable behavior. Routing picks the expert.
SPECIALISTS = {
    "billing": "a billing specialist who handles payments, refunds, and invoices",
    "technical": "a technical support engineer who solves errors and how-to questions",
    "sales": "a sales representative who explains plans, pricing, and upgrades",
    "general": "a friendly general support agent who handles everything else",
}


# ── CONCEPT 2 (+6): Node that classifies the ticket into structured data ──
def classify_ticket(state: SupportState) -> dict:
    """Use structured output to triage the ticket into a validated object."""
    llm = get_llm()
    # `with_structured_output` forces the model to return a TicketClassification,
    # never a raw string we have to hope is valid JSON (concept 6).
    classifier = llm.with_structured_output(TicketClassification)
    result: TicketClassification = classifier.invoke(
        [
            SystemMessage(content="You triage customer support tickets. Be decisive."),
            HumanMessage(
                content=(
                    f"Customer tier: {state.get('customer_tier', 'Free')}\n"
                    f"Message: {state['customer_message']}"
                )
            ),
        ]
    )
    return {
        # Store a plain dict (not the raw Pydantic object) so the checkpointer
        # can serialize state cleanly — a good habit: keep state JSON-safe.
        "classification": result.model_dump(),
        "route": result.category,
        # High severity, an angry customer, or the model's own flag → get a human.
        "escalation_required": (
            result.escalate
            or result.severity in ("high", "critical")
            or result.sentiment == "angry"
        ),
        "history": [f"classified as {result.category}/{result.severity} "
                    f"(confidence {result.confidence:.0%})"],
    }


# ── CONCEPT 4: The router — a plain function returning the next node's name ──
def route_ticket(state: SupportState) -> str:
    """Conditional edge: send the ticket to the right specialist."""
    return state.get("route", "general")


# ── CONCEPT 2 (+5): Retrieval node wrapping the retrieve→rerank→filter pipeline
def retrieve_context(state: SupportState) -> dict:
    """Pull the most relevant knowledge-base articles for this ticket."""
    articles = retrieve_rerank_filter(state["customer_message"])
    titles = [a["title"] for a in articles]
    return {
        "retrieved_articles": articles,
        "history": [f"retrieved {len(articles)} article(s): {', '.join(titles)}"],
    }


def _make_specialist(route: str):
    """Factory that builds a specialist generation node for a given persona."""
    persona = SPECIALISTS[route]

    def specialist_node(state: SupportState) -> dict:
        llm = get_llm(temperature=0.3)
        # Ground the answer in retrieved articles (concept 5 feeding the LLM).
        context = "\n\n".join(
            f"[{a['title']}] {a['text']}" for a in state.get("retrieved_articles", [])
        ) or "No specific articles found; answer from general knowledge."

        messages = [
            SystemMessage(
                content=(
                    f"You are {persona}. Write a concise, warm reply (3-5 sentences). "
                    f"Use ONLY the knowledge base below; if it lacks the answer, say a "
                    f"human will follow up. Customer tier: {state.get('customer_tier')}."
                    f"\n\n=== KNOWLEDGE BASE ===\n{context}"
                )
            ),
            HumanMessage(content=state["customer_message"]),
        ]
        draft = llm.invoke(messages).content
        return {
            "draft_response": draft,
            "history": [f"{route} specialist drafted a response"],
        }

    specialist_node.__name__ = f"{route}_specialist"
    return specialist_node


# Concrete specialist nodes, one per route (the "team of experts").
billing_specialist = _make_specialist("billing")
technical_specialist = _make_specialist("technical")
sales_specialist = _make_specialist("sales")
general_specialist = _make_specialist("general")


# ── CONCEPT 10: Human-in-the-loop ────────────────────────────────
def human_review(state: SupportState) -> dict:
    """
    Pause the graph and wait for a human decision.

    `interrupt(...)` freezes execution and surfaces its payload to whoever is
    driving the graph (CLI, web UI, ticketing system). The graph resumes only
    when we send a `Command(resume=...)` back in — the human's verdict then
    flows into state like any other value. This is how you get AI + human
    collaboration instead of blind automation.
    """
    decision = interrupt(
        {
            "reason": "This ticket was flagged for human review before sending.",
            "ticket_id": state.get("ticket_id"),
            "classification": state.get("classification"),
            "draft_response": state.get("draft_response"),
        }
    )
    # `decision` is whatever the reviewer resumed with, e.g.
    #   {"action": "approve"} | {"action": "edit", "text": "..."} | {"action": "reject"}
    action = (decision or {}).get("action", "approve")
    if action == "edit":
        return {
            "final_response": decision.get("text", state.get("draft_response", "")),
            "human_decision": "edit",
            "history": ["human edited and approved the response"],
        }
    if action == "reject":
        return {
            "final_response": "This ticket needs a specialist. A human agent will reply shortly.",
            "human_decision": "reject",
            "history": ["human rejected the draft"],
        }
    return {
        "final_response": state.get("draft_response", ""),
        "human_decision": "approve",
        "history": ["human approved the draft as-is"],
    }


# ── Auto-finalize when no human review is needed ─────────────────
def finalize(state: SupportState) -> dict:
    """Ship the draft as the final response (the auto-approve path)."""
    return {
        "final_response": state.get("draft_response", ""),
        "history": ["auto-approved and sent (no escalation needed)"],
    }


def needs_human(state: SupportState) -> str:
    """Conditional edge: branch to human review or auto-finalize."""
    return "human_review" if state.get("escalation_required") else "finalize"
