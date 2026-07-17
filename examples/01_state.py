"""
CONCEPT 1 — State: the shared memory of an agent.

Run:  python examples/01_state.py

Takeaway: nodes don't pass arguments to each other — they all read from and
write to ONE shared, typed state object. That shared context is what makes a
multi-step agent debuggable and predictable.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow `python examples/..`

from support_agent.state import new_ticket_state

# Start with just the incoming ticket.
state = new_ticket_state(
    ticket_id="TKT-9012",
    customer_message="My payment failed twice this week!",
    customer_tier="Enterprise",
)
print("Initial state:")
for k, v in state.items():
    print(f"  {k:20} = {v!r}")

# A "node" is just something that returns a PARTIAL update to the state.
# LangGraph merges these back in; here we merge by hand to see the idea.
def fake_retrieve(s):    return {"retrieved_articles": ["Payment Retry Policy"]}
def fake_classify(s):    return {"route": "billing", "escalation_required": True}
def fake_draft(s):       return {"draft_response": "Please retry the payment under Settings > Billing."}

print("\nWatch the SAME state grow as each node contributes:")
for node in (fake_retrieve, fake_classify, fake_draft):
    update = node(state)
    state.update(update)  # LangGraph does this merge for you inside a real graph
    print(f"  after {node.__name__:14} -> {update}")

print("\nFinal shared state:")
for k, v in state.items():
    print(f"  {k:20} = {v!r}")

print("\nWithout state, step 3 would have NO idea what steps 1 and 2 found.")
