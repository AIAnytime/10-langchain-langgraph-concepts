"""
SupportGraph — interactive CLI demo.

Runs the full customer-support agent and shows ALL 10 concepts working
together, live, in your terminal:

  1 State      2 Nodes     3 Graph        4 Routing     5 Retrieval
  6 Structured 7 Streaming 8 Memory       9 Checkpoint  10 Human-in-the-loop

Usage:
    python cli.py                       # pick from example tickets
    python cli.py "my payment failed"   # run one ticket directly
"""
import sys
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from langgraph.types import Command

from support_agent.config import has_api_key, DEFAULT_MODEL
from support_agent.graph import build_graph
from support_agent.state import new_ticket_state

console = Console()

EXAMPLES = [
    ("A calm billing question", "Free", "How do I get a refund on my monthly plan?"),
    ("An angry billing issue", "Enterprise", "URGENT! I was charged twice and no one replied. Fix this now!"),
    ("A technical question", "Pro", "I keep getting HTTP 429 errors from your API. What do I do?"),
    ("A sales question", "Free", "What's the difference between Pro and Enterprise pricing?"),
]

# Which concept each node demonstrates — printed live so viewers can follow along.
NODE_CONCEPTS = {
    "classify": "6 Structured output + 4 Routing decision",
    "retrieve": "5 Retrieval (retrieve → rerank → filter)",
    "billing": "2 Node + 4 Routed to billing specialist",
    "technical": "2 Node + 4 Routed to technical specialist",
    "sales": "2 Node + 4 Routed to sales specialist",
    "general": "2 Node + 4 Routed to general specialist",
    "human_review": "10 Human-in-the-loop pause",
    "finalize": "auto-approved (no human needed)",
}


def show_update(node: str, update: dict) -> None:
    """Concept 7 (streaming): print progress as each node finishes."""
    concept = NODE_CONCEPTS.get(node, "")
    console.print(f"[bold cyan]▶ {node}[/]  [dim]{concept}[/]")
    if node == "classify" and update.get("classification"):
        c = update["classification"]
        console.print(
            f"   → [yellow]{c['category']}[/] · severity [yellow]{c['severity']}[/] · "
            f"sentiment [yellow]{c['sentiment']}[/] · confidence {c['confidence']:.0%}"
        )
    if node == "retrieve":
        for a in update.get("retrieved_articles", []):
            console.print(f"   → 📄 {a['title']}")
    if update.get("draft_response"):
        console.print(Panel(update["draft_response"], title="Draft response", border_style="green"))


def handle_human_review(interrupt_value: dict):
    """Concept 10: pause, ask the human, return a resume Command."""
    console.print(Panel(
        f"[bold]{interrupt_value.get('reason','Review required')}[/]\n\n"
        f"[dim]Draft:[/] {interrupt_value.get('draft_response','')}",
        title="🧑 HUMAN REVIEW NEEDED", border_style="red",
    ))
    choice = Prompt.ask(
        "Your decision", choices=["approve", "edit", "reject"], default="approve"
    )
    if choice == "edit":
        text = Prompt.ask("Enter the edited response")
        return Command(resume={"action": "edit", "text": text})
    return Command(resume={"action": choice})


def run_ticket(app, message: str, tier: str) -> None:
    thread_id = f"ticket-{uuid.uuid4().hex[:8]}"          # concepts 8 & 9: memory/checkpoint key
    config = {"configurable": {"thread_id": thread_id}}
    console.rule(f"[bold]New ticket[/]  (thread {thread_id})")
    console.print(f"[dim]Tier:[/] {tier}   [dim]Message:[/] {message}\n")

    state = new_ticket_state(thread_id, message, tier)

    # Concept 7: stream node-by-node updates instead of waiting for the end.
    stream = app.stream(state, config, stream_mode="updates")
    while True:
        interrupted = False
        for chunk in stream:
            if "__interrupt__" in chunk:                  # concept 10: graph paused
                cmd = handle_human_review(chunk["__interrupt__"][0].value)
                stream = app.stream(cmd, config, stream_mode="updates")  # resume
                interrupted = True
                break
            for node, update in chunk.items():
                show_update(node, update)
        if not interrupted:
            break

    # Concept 1 & 8: the final shared state holds the whole audit trail.
    final = app.get_state(config).values
    console.print(Panel(final.get("final_response", "(no response)"),
                        title="✅ FINAL RESPONSE SENT", border_style="bold green"))
    table = Table(title="Agent memory / audit trail (state.history)", show_header=False)
    for i, event in enumerate(final.get("history", []), 1):
        table.add_row(str(i), event)
    console.print(table)


def main() -> None:
    console.print(Panel.fit(
        "[bold]SupportGraph[/] — 10 LangChain/LangGraph concepts, live\n"
        f"[dim]Model via OpenRouter: {DEFAULT_MODEL}[/]",
        border_style="magenta",
    ))
    if not has_api_key():
        console.print("[red]OPENROUTER_API_KEY not set.[/] Copy .env.example to .env and add your key.")
        sys.exit(1)

    app = build_graph()

    if len(sys.argv) > 1:                                 # ticket passed on the command line
        run_ticket(app, " ".join(sys.argv[1:]), tier="Pro")
        return

    console.print("\n[bold]Pick an example ticket:[/]")
    for i, (label, tier, msg) in enumerate(EXAMPLES, 1):
        console.print(f"  [cyan]{i}[/] {label} [dim]({tier})[/] — {msg}")
    console.print("  [cyan]c[/] custom ticket")
    choice = Prompt.ask("Choice", default="1")

    if choice == "c":
        msg = Prompt.ask("Ticket message")
        tier = Prompt.ask("Customer tier", choices=["Free", "Pro", "Enterprise"], default="Free")
    else:
        _, tier, msg = EXAMPLES[int(choice) - 1]
    run_ticket(app, msg, tier)


if __name__ == "__main__":
    main()
