"""
SupportGraph — Streamlit web UI.

A visual, clickable demo of the same agent as cli.py. Great for a screen
recording: you can watch the graph route, retrieve, stream progress, and then
PAUSE for your approval before the reply is sent.

Run:
    streamlit run app.py
"""
import uuid

import streamlit as st
from langgraph.types import Command

from support_agent.config import DEFAULT_MODEL, has_api_key
from support_agent.graph import build_graph
from support_agent.state import new_ticket_state

st.set_page_config(page_title="SupportGraph — 10 LangGraph Concepts", page_icon="🤖", layout="wide")

CONCEPTS = [
    "1 State — shared typed memory", "2 Nodes — functions", "3 Graph — not a chain",
    "4 Routing — pick a specialist", "5 Retrieval — retrieve/rerank/filter",
    "6 Structured outputs — Pydantic", "7 Streaming — live progress",
    "8 Memory — audit trail", "9 Checkpointing — durable/resumable",
    "10 Human-in-the-loop — approval gate",
]
NODE_CONCEPT = {
    "classify": "6 Structured output + 4 Routing", "retrieve": "5 Retrieval pipeline",
    "billing": "4 Billing specialist", "technical": "4 Technical specialist",
    "sales": "4 Sales specialist", "general": "4 General specialist",
    "human_review": "10 Human-in-the-loop", "finalize": "auto-approved",
}


@st.cache_resource
def get_app():
    """Build the graph once; its MemorySaver persists across Streamlit reruns."""
    return build_graph()


def drive(app, payload, config, log):
    """Run/resume the graph, streaming node updates into `log`. Returns interrupt or None."""
    for chunk in app.stream(payload, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            return chunk["__interrupt__"][0].value
        for node, update in chunk.items():
            log.markdown(f"**▶ {node}** — _{NODE_CONCEPT.get(node, '')}_")
            if node == "classify" and update.get("classification"):
                c = update["classification"]
                log.info(f"Classified: **{c['category']}** · severity **{c['severity']}** · "
                         f"sentiment **{c['sentiment']}** · confidence {c['confidence']:.0%}")
            if node == "retrieve":
                for a in update.get("retrieved_articles", []):
                    log.markdown(f"&nbsp;&nbsp;📄 {a['title']}")
            if update.get("draft_response"):
                log.success(update["draft_response"])
    return None


# ── Sidebar: the concept checklist ──────────────────────────────
with st.sidebar:
    st.header("🤖 SupportGraph")
    st.caption(f"OpenRouter model: `{DEFAULT_MODEL}`")
    st.subheader("10 concepts in this demo")
    for c in CONCEPTS:
        st.markdown(f"- {c}")

st.title("Customer Support Agent")
st.caption("One practical use case demonstrating all 10 LangChain/LangGraph concepts.")

if not has_api_key():
    st.error("`OPENROUTER_API_KEY` not set. Copy `.env.example` to `.env` and add your key.")
    st.stop()

app = get_app()
ss = st.session_state
ss.setdefault("phase", "idle")   # idle | awaiting_human | done

# ── Ticket input ────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
message = col1.text_area("Customer message", "URGENT! I was charged twice and no one replied!", height=90)
tier = col2.selectbox("Customer tier", ["Free", "Pro", "Enterprise"], index=2)

if st.button("▶ Run agent", type="primary", disabled=ss.phase == "awaiting_human"):
    ss.thread_id = f"ticket-{uuid.uuid4().hex[:8]}"
    ss.config = {"configurable": {"thread_id": ss.thread_id}}
    with st.status("Running the graph...", expanded=True) as status:
        interrupt = drive(app, new_ticket_state(ss.thread_id, message, tier), ss.config, st)
        if interrupt:
            ss.phase, ss.interrupt = "awaiting_human", interrupt
            status.update(label="Paused for human review (concept 10)", state="error")
        else:
            ss.phase = "done"
            status.update(label="Done — auto-approved", state="complete")

# ── Concept 10: human review gate (survives Streamlit reruns via session_state) ──
if ss.phase == "awaiting_human":
    st.warning("🧑 **Human review needed** before this reply is sent.")
    st.markdown("**Draft response:**")
    st.info(ss.interrupt.get("draft_response", ""))
    edited = st.text_area("Edit before approving (optional)", ss.interrupt.get("draft_response", ""))
    a, b, c = st.columns(3)
    resume = None
    if a.button("✅ Approve"):
        resume = Command(resume={"action": "approve"})
    if b.button("✏️ Approve edit"):
        resume = Command(resume={"action": "edit", "text": edited})
    if c.button("❌ Reject"):
        resume = Command(resume={"action": "reject"})
    if resume is not None:
        with st.status("Resuming from checkpoint...", expanded=True) as status:
            drive(app, resume, ss.config, st)          # concept 9: resume same thread
            status.update(label="Done", state="complete")
        ss.phase = "done"

# ── Final response + memory/audit trail ─────────────────────────
if ss.phase == "done":
    final = app.get_state(ss.config).values
    st.subheader("✅ Final response sent")
    st.success(final.get("final_response", "(none)"))
    with st.expander("🧠 Agent memory — full audit trail (state.history)", expanded=True):
        for i, event in enumerate(final.get("history", []), 1):
            st.markdown(f"{i}. {event}")
    st.caption(f"Thread `{ss.thread_id}` — state is checkpointed and resumable (concepts 8 & 9).")
