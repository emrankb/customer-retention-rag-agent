"""LangChain Runnable steps for customer lookup, retrieval, and answer generation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.runnables import RunnableLambda

from .assistant import _validate_question, answer_question
from .customer_db import CustomerDB
from .retrieval import search_policies


def make_rag_chain(
    *,
    customer_db: CustomerDB,
    docs_dir: str | Path,
    index_dir: str | Path,
    generate: Callable[[str], str],
    search: Callable = search_policies,
):
    """Compose three Runnable steps; model calls happen only on invoke()."""

    def load_customer(request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("Expected a question request.")
        question = _validate_question(request.get("question"))
        customer_id = request.get("customer_id")
        customer = customer_db.get_context(customer_id)
        return {
            "question": question,
            "customer_id": customer_id,
            "customer": customer,
            "history": request.get("history"),
        }

    def retrieve(state: dict) -> dict:
        passages = (
            search(state["question"], docs_dir, index_dir, k=4)
            if state["customer"] is not None else []
        )
        return {**state, "passages": passages}

    def respond(state: dict):
        # Use the existing answer validation without looking up the same
        # customer or searching the index a second time.
        class LoadedCustomer:
            def get_context(self, _identifier):
                return state["customer"]

        def loaded_passages(_question, _docs_dir, _index_dir, *, k):
            return state["passages"]

        return answer_question(
            state["question"], state["customer_id"],
            customer_db=LoadedCustomer(),
            docs_dir=docs_dir,
            index_dir=index_dir,
            generate=generate,
            search=loaded_passages,
            history=state["history"],
        )

    return (
        RunnableLambda(load_customer)
        | RunnableLambda(retrieve)
        | RunnableLambda(respond)
    )
