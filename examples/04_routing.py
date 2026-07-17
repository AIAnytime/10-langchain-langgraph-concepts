"""
CONCEPT 4 — Routing beats one giant prompt.

Run:  python examples/04_routing.py   (offline keyword demo + optional LLM route)

Takeaway: don't build one "you are a billing + tech + sales + legal expert"
mega-prompt. Route to a small, focused specialist. Shorter prompts = higher
accuracy, lower cost, easier maintenance. Best AI systems are a team of experts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── The naive, cheap router: keyword rules (great to explain the idea) ──
def keyword_router(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("payment", "refund", "invoice", "charge", "billing")):
        return "billing_agent"
    if any(w in q for w in ("error", "429", "bug", "reset", "login", "api")):
        return "technical_agent"
    if any(w in q for w in ("price", "pricing", "plan", "upgrade", "enterprise")):
        return "sales_agent"
    return "general_agent"

tickets = [
    "My payment failed and I want a refund",
    "I keep getting a 429 error from the API",
    "How much does the Enterprise plan cost?",
    "Can you export my data?",
]
print("=== Keyword routing ===")
for t in tickets:
    print(f"  {keyword_router(t):16} <- {t}")

# ── The production router: an LLM classifier with STRUCTURED output (concept 6) ──
from support_agent.config import has_api_key
if has_api_key():
    from support_agent.nodes import classify_ticket
    print("\n=== LLM structured routing (more robust than keywords) ===")
    for t in tickets:
        update = classify_ticket({"customer_message": t, "customer_tier": "Pro"})
        c = update["classification"]  # a JSON-safe dict from the structured classifier
        print(f"  {c['category']:10} sev={c['severity']:8} conf={c['confidence']:.0%} <- {t}")
else:
    print("\n(set OPENROUTER_API_KEY to see the LLM-based router)")
