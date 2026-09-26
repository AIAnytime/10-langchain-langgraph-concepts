# 10 LangChain & LangGraph Concepts Every AI Engineer Should Know

A hands-on tutorial project. One practical use case — a production-style
**customer-support agent** — demonstrates all 10 concepts, live, with runnable
code. Built with **LangGraph + LangChain**, powered by any model via **OpenRouter**.

Companion to the YouTube video/short. Every file is heavily commented so you can
read it on screen.

---

## The 10 concepts → where to see each one

| # | Concept | Standalone example | In the full agent |
|---|---------|--------------------|-------------------|
| 1 | **State** — shared, typed memory | `examples/01_state.py` | `support_agent/state.py` |
| 2 | **Nodes** — a node is just a function | `examples/02_nodes.py` | `support_agent/nodes.py` |
| 3 | **Chains vs Graphs** | `examples/03_chains_vs_graphs.py` | `support_agent/graph.py` |
| 4 | **Routing** beats one giant prompt | `examples/04_routing.py` | `nodes.route_ticket` |
| 5 | **Retrieval** = retrieve → rerank → filter | `examples/05_retrieval.py` | `support_agent/retrieval.py` |
| 6 | **Structured outputs** (Pydantic) | `examples/06_structured_outputs.py` | `support_agent/schemas.py` |
| 7 | **Streaming** — perceived speed | `examples/07_streaming.py` | `cli.py` / `app.py` progress |
| 8 | **Memory** — operational audit trail | `examples/08_memory.py` | `state.history` |
| 9 | **Checkpointing** — durable & resumable | `examples/09_checkpointing.py` | `graph.build_graph()` |
| 10 | **Human-in-the-loop** — approval gate | `examples/10_human_in_the_loop.py` | `nodes.human_review` |

---

## Setup

```bash
# 1. Create + activate a virtual environment (Python 3.11–3.13 recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenRouter key
cp .env.example .env               # then edit .env and paste your key
# Get a key at https://openrouter.ai/keys
```

`.env`:
```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini   # optional; any tool-calling model works
```

---

## Run it

**The 10 concept examples** (run individually — most work offline; 03/06/07 call the model):
```bash
python examples/01_state.py
python examples/05_retrieval.py
python examples/06_structured_outputs.py
# ...through 10
```

**The full agent — CLI demo** (routing + retrieval + streaming + human approval):
```bash
python cli.py                          # pick an example ticket
python cli.py "I was charged twice!"   # or pass your own
```

**The full agent — Web UI demo:**
```bash
streamlit run app.py
```

---

## How the agent works

```
       classify ──▶ retrieve ──▶ [route] ──▶ billing / technical / sales / general
       (concept 6)  (concept 5)  (concept 4)              │
                                                  [escalate?]  (concept 10)
                                              ┌────────────┴────────────┐
                                         human_review              finalize
                                              └────────────┬────────────┘
                                                          END
```

A ticket is triaged into a **validated** classification, routed to the right
**specialist**, answered using **retrieved** knowledge-base articles, and — if
it's high-severity or the customer is upset — **paused for a human** to approve,
edit, or reject before sending. Every run is **checkpointed** and keeps a full
**memory** trail.

## Project structure

```
support_agent/
  config.py          OpenRouter LLM wiring
  state.py           1. State  / 8. Memory
  schemas.py         6. Structured outputs
  knowledge_base.py  demo data
  retrieval.py       5. Retrieve → rerank → filter
  nodes.py           2. Nodes / 4. Routing / 10. Human-in-the-loop
  graph.py           3. Graph / 9. Checkpointing
examples/            10 standalone, runnable concept demos
cli.py               full agent — terminal demo
app.py               full agent — Streamlit web UI
```

## Notes

- **OpenRouter** is OpenAI-compatible, so we reuse LangChain's `ChatOpenAI` and
  only change `base_url`. Swap in any provider the same way.
- The knowledge base is in-memory (TF-IDF) so the retrieval demo runs offline.
  In production, replace `retrieval.retrieve()` with a real vector store — the
  rerank/filter stages stay the same.
- `.env` is git-ignored. Never commit your API key.

## License

Proprietary — all rights reserved. This code is published for viewing and evaluation only; no use, copying, modification, redistribution, commercial use, or use as AI/ML training data without written permission. See [LICENSE](LICENSE). Commercial licensing: aianytime07@gmail.com · sonu@aianytime.net.
