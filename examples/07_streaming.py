"""
CONCEPT 7 — Streaming changes the user's perception of speed.

Run:  python examples/07_streaming.py   (needs an API key)

Takeaway: 15 seconds of a blank screen feels slow; a response that starts
flowing immediately feels fast — even at the same total time. Stream tokens,
and for multi-step workflows, stream progress at each stage.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support_agent.config import has_api_key, get_llm

if not has_api_key():
    print("Set OPENROUTER_API_KEY in .env to run this example.")
    raise SystemExit(0)

# ── 1) Token streaming: print each chunk as it arrives. ──
print("=== Streaming tokens (word-by-word) ===")
for chunk in get_llm().stream("Explain what an API rate limit is, in 2 sentences."):
    print(chunk.content, end="", flush=True)
print("\n")

# ── 2) Progress streaming: for a workflow, push a status per stage. ──
print("=== Streaming workflow progress ===")
stages = [
    "Analyzing the ticket...",
    "Retrieving knowledge-base articles...",
    "Drafting a response...",
    "Validating before sending...",
]
for msg in stages:
    print("  •", msg, flush=True)
    time.sleep(0.4)  # stand-in for real work happening in each node
print("  ✓ Done. Perceived performance often matters more than actual performance.")
