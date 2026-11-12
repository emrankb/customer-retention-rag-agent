"""Gradio interface for customer questions and cited policy answers."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .assistant import answer_question
from .customer_db import CustomerDB


def make_handler(
    *,
    customer_db: CustomerDB,
    docs_dir: str | Path,
    index_dir: str | Path,
    generate: Callable[[str], str],
    answer=answer_question,
):
    """Return a Gradio-compatible callback with no hidden model calls on import."""
    def handle(customer_id: str, question: str) -> str:
        if not isinstance(customer_id, str) or not customer_id.strip():
            return "Enter a customer ID."
        if not isinstance(question, str) or not question.strip():
            return "Enter a question."
        try:
            result = answer(
                question, customer_id.strip(), customer_db=customer_db,
                docs_dir=docs_dir, index_dir=index_dir, generate=generate,
            )
        except ValueError as exc:
            return f"Invalid input: {exc}"
        except Exception:
            # Do not show customer data, model output, or internal paths in the UI.
            return "The request could not be completed. Check the application logs."
        citations = "\n".join(
            f"- {item['marker']} {item['source']} "
            f"(lines {item['line_start']}-{item['line_end']})"
            for item in result.citations
        )
        return result.text + ("\n\nSources:\n" + citations if citations else "")
    return handle


def create_app(
    *,
    customer_db: CustomerDB,
    docs_dir: str | Path,
    index_dir: str | Path,
    generate: Callable[[str], str],
):
    """Create the UI only when Gradio is installed and explicitly requested."""
    import gradio as gr

    handler = make_handler(
        customer_db=customer_db, docs_dir=docs_dir,
        index_dir=index_dir, generate=generate,
    )
    return gr.Interface(
        fn=handler,
        inputs=[
            gr.Textbox(label="Customer ID", placeholder="Enter a customer ID"),
            gr.Textbox(label="Question", lines=3, placeholder="Ask about a bank policy"),
        ],
        outputs=gr.Markdown(label="Answer"),
        title="Customer retention assistant",
        description="Answers use available customer records and policy excerpts. "
                    "Account decisions require authorized review.",
        flagging_mode="never",
    )
