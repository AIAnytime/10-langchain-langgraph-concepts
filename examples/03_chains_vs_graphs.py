"""
CONCEPT 3 — Chains vs Graphs.

Run:  python examples/03_chains_vs_graphs.py   (chain part needs an API key)

Takeaway: a CHAIN is linear (prompt → LLM → parser) and perfect for simple
flows. A GRAPH adds branching, loops, and pauses — which real workflows need.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support_agent.config import has_api_key, get_llm

# ── A CHAIN: LangChain's `|` pipes components left-to-right. Linear only. ──
print("=== CHAIN (linear) ===")
if has_api_key():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(
        "Summarize this support ticket in 5 words: {ticket}"
    )
    chain = prompt | get_llm() | StrOutputParser()   # prompt → LLM → parser
    print("summary:", chain.invoke({"ticket": "My payment failed twice this week."}))
else:
    print("(skipped — set OPENROUTER_API_KEY to run the live chain)")

# ── A GRAPH: same idea, but edges can branch/converge. Shape it as a diagram. ──
print("\n=== GRAPH (branching) ===")
from support_agent.graph import build_graph

app = build_graph()
# LangGraph can render its own structure — great for a video slide.
print(app.get_graph().draw_ascii())
print(
    "A chain could never express 'route to a specialist, then maybe pause for a\n"
    "human'. The graph does it declaratively."
)
