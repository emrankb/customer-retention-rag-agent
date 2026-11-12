"""Gradio interface for customer questions and cited policy answers."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .assistant import answer_question
from .customer_db import CustomerDB
from .retrieval import search_policies


def make_handler(
    *,
    customer_db: CustomerDB,
    docs_dir: str | Path,
    index_dir: str | Path,
    generate: Callable[[str], str],
    answer=answer_question,
    search=search_policies,
):
    """Return a Gradio-compatible callback with no hidden model calls on import."""
    def handle(customer_id: str, question: str) -> str:
        if not isinstance(customer_id, str) or not customer_id.strip():
            return "Enter a customer ID."
        if not isinstance(question, str) or not question.strip():
            return "Enter a question."
        identifier = customer_id.strip()
        if not identifier.isascii() or not identifier.isdecimal():
            return "Invalid input: Customer ID must contain only digits."
        if len(question) > 1000 or "\x00" in question:
            return "Invalid input: Question must be 1–1000 characters."
        try:
            result = answer(
                question, identifier, customer_db=customer_db,
                docs_dir=docs_dir, index_dir=index_dir, generate=generate,
                search=search,
            )
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


def make_chat_handler(chain):
    """Handle one turn and keep history separate for each customer ID."""
    def handle(customer_id: str, question: str, state: dict | None):
        if not isinstance(customer_id, str) or not customer_id.strip():
            return [], "", {"customer_id": None, "messages": []}
        identifier = customer_id.strip()
        if not identifier.isascii() or not identifier.isdecimal():
            return ([{"role": "assistant", "content": "Enter a valid customer ID."}],
                    "", {"customer_id": None, "messages": []})
        if not isinstance(question, str) or not question.strip():
            return [], "", {"customer_id": identifier, "messages": []}
        if len(question) > 1000 or "\x00" in question:
            return ([{"role": "assistant", "content": "Question must be 1–1000 characters."}],
                    "", {"customer_id": identifier, "messages": []})
        previous = (state.get("messages", []) if isinstance(state, dict)
                    and state.get("customer_id") == identifier else [])
        try:
            result = chain.invoke({
                "customer_id": identifier, "question": question,
                "history": previous[-6:],
            })
            sources = "\n".join(
                f"- {c['marker']} {c['source']} "
                f"(lines {c['line_start']}-{c['line_end']})"
                for c in result.citations
            )
            answer = result.text + ("\n\nSources:\n" + sources if sources else "")
        except Exception:
            answer = "The request could not be completed. Check the application logs."
        messages = (previous + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])[-12:]
        return messages, "", {"customer_id": identifier, "messages": messages}

    return handle


def create_app(
    *,
    customer_db: CustomerDB,
    docs_dir: str | Path,
    index_dir: str | Path,
    generate: Callable[[str], str],
    search=search_policies,
    chain=None,
):
    """Create a Gradio chat interface with session-scoped customer history."""
    import gradio as gr
    from .rag_chain import make_rag_chain

    if chain is None:
        chain = make_rag_chain(
            customer_db=customer_db, docs_dir=docs_dir,
            index_dir=index_dir, generate=generate, search=search,
        )
    handle = make_chat_handler(chain)
    with gr.Blocks(title="Customer retention assistant") as app:
        gr.Markdown("# Customer retention assistant")
        gr.Markdown(
            "Answers use customer records and policy documents. "
            "Account decisions require authorized review."
        )
        customer_id = gr.Textbox(label="Customer ID")
        conversation = gr.Chatbot(label="Conversation")
        question = gr.Textbox(label="Question", lines=2)
        submit = gr.Button("Send")
        session = gr.State({"customer_id": None, "messages": []})
        controls = [customer_id, question, session]
        outputs = [conversation, question, session]
        submit.click(handle, inputs=controls, outputs=outputs)
        question.submit(handle, inputs=controls, outputs=outputs)

        # Changing a customer ID must not carry over another customer's chat.
        customer_id.change(
            lambda: ([], {"customer_id": None, "messages": []}),
            inputs=[], outputs=[conversation, session],
        )
    return app
