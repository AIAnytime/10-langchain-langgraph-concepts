"""
CONCEPT 10 — Human-in-the-loop collaboration.

Run:  python examples/10_human_in_the_loop.py   (runs offline — no API key needed)

Takeaway: most enterprise AI shouldn't be fully autonomous. The best systems
pause at high-stakes moments, let a human decide, then continue. LangGraph's
`interrupt()` + `Command(resume=...)` make this a first-class pattern.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class ReviewState(TypedDict):
    risk_score: float
    action: str

def analyze(s):        return {"risk_score": 0.9}   # AI analysis produced high risk
def review_gate(s):
    if s["risk_score"] > 0.8:
        # High risk → hand control to a human and WAIT for their verdict.
        human = interrupt({"risk_score": s["risk_score"],
                           "question": "Approve this action?"})
        return {"action": human}
    return {"action": "auto-approved"}              # low risk → proceed automatically

g = StateGraph(ReviewState)
g.add_node("analyze", analyze); g.add_node("review", review_gate)
g.add_edge(START, "analyze"); g.add_edge("analyze", "review"); g.add_edge("review", END)
app = g.compile(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "case-7"}}

# 1) Run until the graph pauses for a human.
app.invoke({}, cfg)
paused = app.get_state(cfg)
print("Graph PAUSED for human review.")
print("  interrupt payload:", paused.tasks[0].interrupts[0].value)

# 2) A human reviews and responds. Resume the graph with their decision.
human_says = "approve"     # imagine a reviewer clicked a button in a UI
result = app.invoke(Command(resume=human_says), cfg)
print(f"\nHuman said {human_says!r} -> graph resumed.")
print("  final action:", result["action"])
print("\nAI + human, not AI instead of human — that's the reliable pattern.")
