"""
CONCEPT 6 — Structured outputs avoid costly failures.

Run:  python examples/06_structured_outputs.py   (needs an API key)

Takeaway: "return valid JSON" in a prompt is a promise the model can break —
one missing comma crashes your parser. Bind a Pydantic schema instead and get a
validated object every time. Schema should be mandatory when other systems
consume the output.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support_agent.config import has_api_key, get_llm
from support_agent.schemas import TicketClassification

if not has_api_key():
    print("Set OPENROUTER_API_KEY in .env to run this example.")
    raise SystemExit(0)

# `.with_structured_output(Schema)` makes the model fill in a tool/JSON schema.
# The result is a validated Pydantic object — never a raw, fragile string.
structured_llm = get_llm().with_structured_output(TicketClassification)

ticket = "URGENT!! I was charged twice and nobody is responding. This is unacceptable."
result = structured_llm.invoke(ticket)

print("Input ticket:\n ", ticket, "\n")
print("Validated, typed result (a TicketClassification object):")
print("  category  :", result.category)
print("  severity  :", result.severity)
print("  sentiment :", result.sentiment)
print("  escalate  :", result.escalate)
print("  confidence:", f"{result.confidence:.0%}")
print("  reason    :", result.reason)

# Because it's validated, downstream code can trust the types.
assert result.category in {"billing", "technical", "sales", "general"}
print("\n✓ Guaranteed shape — safe to feed into routing, DBs, or other systems.")
