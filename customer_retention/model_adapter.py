"""Explicit adapter for a configured LangChain chat model."""
from __future__ import annotations

import os


def create_generator(*, model_name: str | None = None):
    """Build a callable; requests happen only when it is called."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment before starting.")
    model = model_name or os.environ.get("RETENTION_MODEL", "gpt-4o-mini")
    if not model.strip():
        raise ValueError("Model name cannot be blank.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Install requirements-app.txt to use the chat model.") from exc
    chat = ChatOpenAI(model=model, temperature=0)

    def generate(prompt: str) -> str:
        result = chat.invoke(prompt)
        if not isinstance(result.content, str):
            raise ValueError("The model returned a non-text response.")
        return result.content

    return generate
