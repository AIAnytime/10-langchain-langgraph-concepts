"""
CONCEPT 9 — Checkpointing supports long-running workflows.

Run:  python examples/09_checkpointing.py   (runs offline — no API key needed)

Takeaway: real systems fail — timeouts, outages, reviews that take days.
Without checkpoints a failure means starting over. With checkpoints you resume
from exactly where you stopped. Nothing already done is recomputed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# A loan-approval-style flow: collect → assess → (pause) → decide.
class LoanState(TypedDict):
    applicant: str
    risk_score: float
    decision: str

def collect_docs(s):    print("  [node] collecting documents...");  return {}
def assess_risk(s):     print("  [node] assessing risk...");       return {"risk_score": 0.72}
def human_approval(s):
    # Pause here — a human underwriter may take hours or days.
    verdict = interrupt({"applicant": s["applicant"], "risk_score": s["risk_score"]})
    return {"decision": verdict}
def finalize(s):        print("  [node] finalizing decision...");  return {}

g = StateGraph(LoanState)
for name, fn in [("collect", collect_docs), ("assess", assess_risk),
                 ("approve", human_approval), ("final", finalize)]:
    g.add_node(name, fn)
g.add_edge(START, "collect"); g.add_edge("collect", "assess")
g.add_edge("assess", "approve"); g.add_edge("approve", "final"); g.add_edge("final", END)

# MemorySaver persists every checkpoint. thread_id names this particular run.
app = g.compile(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "loan-123"}}

print("--- Day 1: start the workflow ---")
app.invoke({"applicant": "Acme Corp"}, cfg)
snap = app.get_state(cfg)
print(f"  paused at: {snap.next}  (waiting for a human)")
print("  ...process could crash or shut down here; the checkpoint survives...")

print("\n--- Day 3: resume from the saved checkpoint, no rework ---")
app.invoke(Command(resume="approved"), cfg)   # same thread_id → continues
print("  final decision:", app.get_state(cfg).values["decision"])
print("\nEverything before the pause was NOT recomputed — that's checkpointing.")
