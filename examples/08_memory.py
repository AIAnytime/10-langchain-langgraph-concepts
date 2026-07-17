"""
CONCEPT 8 — Memory is more than chat history.

Run:  python examples/08_memory.py   (runs offline — no API key needed)

Takeaway: real agents keep OPERATIONAL memory — decisions made, tools called,
documents retrieved, current state of the world — not just a list of messages.
That memory is what lets an agent stay consistent across many steps.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operator import add
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver


# The `Annotated[list, add]` reducer APPENDS to memory instead of overwriting —
# so `history` becomes a running audit trail of everything the agent did.
class AgentState(TypedDict):
    repository: str
    history: Annotated[list, add]

def edit_files(s):   return {"history": ["edited payment.service.ts, retry.handler.ts"]}
def run_tests(s):    return {"history": ["tests failed: payment_retry.spec.ts"]}
def open_pr(s):      return {"history": ["opened PR #482 with the fix"]}

graph = StateGraph(AgentState)
for name, fn in [("edit", edit_files), ("test", run_tests), ("pr", open_pr)]:
    graph.add_node(name, fn)
graph.add_edge(START, "edit"); graph.add_edge("edit", "test")
graph.add_edge("test", "pr");  graph.add_edge("pr", END)

# A checkpointer gives us THREAD memory: state persists under a thread_id and
# survives across separate invocations (like a user coming back tomorrow).
app = graph.compile(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "coding-agent-42"}}

final = app.invoke({"repository": "payments-api", "history": []}, cfg)
print("Operational memory the agent accumulated:")
for i, event in enumerate(final["history"], 1):
    print(f"  {i}. {event}")

# The memory is still there later — fetch it by thread_id without re-running.
print("\nRecall by thread_id (persistent thread memory):")
print("  repository :", app.get_state(cfg).values["repository"])
print("  events kept:", len(app.get_state(cfg).values["history"]))
print("\nThis is what turns isolated calls into an intelligent, consistent workflow.")
