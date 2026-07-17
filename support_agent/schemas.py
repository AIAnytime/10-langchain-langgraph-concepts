"""
CONCEPT 6 — Structured outputs

"Return valid JSON" in a prompt is a promise the model can break — a single
missing comma crashes your parser and takes the whole workflow down with it.

The fix is to make the *schema* mandatory. We define a Pydantic model and hand
it to `llm.with_structured_output(...)`. LangChain turns it into a tool/JSON
schema, the model is forced to fill it in, and we get back a validated Python
object every time — never a raw, possibly-broken string.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TicketClassification(BaseModel):
    """Structured triage result for an incoming support ticket."""

    category: Literal["billing", "technical", "sales", "general"] = Field(
        description="The primary topic of the customer's request."
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="How urgent/impactful the issue is for the customer."
    )
    sentiment: Literal["angry", "frustrated", "neutral", "happy"] = Field(
        description="The emotional tone detected in the customer's message."
    )
    escalate: bool = Field(
        description="True if a human agent should review before replying."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in this classification, from 0 to 1.",
    )
    reason: str = Field(
        description="One short sentence explaining the classification."
    )
