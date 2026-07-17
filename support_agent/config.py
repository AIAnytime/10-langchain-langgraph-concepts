"""
Configuration + LLM factory.

Everything talks to the model through OpenRouter (https://openrouter.ai),
which exposes an OpenAI-compatible API. That means we can reuse LangChain's
battle-tested `ChatOpenAI` client and simply point it at OpenRouter's base URL.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load variables from .env into the process environment exactly once.
load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Default model: cheap, fast, and supports tool calling (needed for concept 6).
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def has_api_key() -> bool:
    """True when an OpenRouter key is configured — lets demos degrade gracefully."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


@lru_cache(maxsize=None)
def get_llm(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """
    Build (and cache) a ChatOpenAI client wired to OpenRouter.

    Because OpenRouter is OpenAI-compatible, the ONLY differences from a normal
    OpenAI setup are `base_url` and the API key. Everything downstream —
    `.invoke()`, `.stream()`, `.with_structured_output()` — works unchanged.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://openrouter.ai/keys"
        )

    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        # OpenRouter likes these optional headers for attribution / rankings.
        default_headers={
            "HTTP-Referer": "https://github.com/AIAnytime/10-langchain-langgraph-concepts",
            "X-Title": "LangGraph 10 Concepts Tutorial",
        },
    )
