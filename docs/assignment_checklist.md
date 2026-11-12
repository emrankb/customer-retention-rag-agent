# Assignment requirements check

The two-page capstone asks for six tasks. This check describes the implementation
as of the Runnable/chat-history milestone; it is not a grading result.

| Assignment task | Implementation and evidence | Status |
| --- | --- | --- |
| 1. Organize customer data, policy documents, validate text encoding, create folders | `CustomerDB`, `source_checks.py`, `data/`, `docs/`, `index/`; synthetic checks and supplied-file inspection | Implemented; private-file integration still needs local run |
| 2. LangChain loading, splitting, Hugging Face embeddings, FAISS | `retrieval.py`, index build script, integration tests | Implemented; actual semantic quality not yet measured |
| 3. CustomerDB and get_context; optional sentiment | `customer_db.py`, both supplied CSV formats, sentiment support | Implemented |
| 4. LangChain Runnable RAG chain and configured ChatOpenAI/ChatOllama | `rag_chain.py` combines lookup, retrieval, and answer generation using RunnableLambda; `model_adapter.py` provides ChatOpenAI | Implemented; live API calls not performed in CI |
| 5. Gradio chat, citations, context errors, history | `interface.py` with session-scoped messages and customer-ID reset; citation/source display | Implemented; browser interaction needs local confirmation |
| 6. Sample queries and quality evaluation (grounding, personalization, tone, coherence) | Assignment sample questions, response capture command, review rubric, offline tests and retrieval scoring | Capture and review tooling implemented; real-model ratings still incomplete |

The task list mentions Model Context and Tooling Protocols in the course title,
but no task explicitly asks for an MCP server. It should not be added merely
to match the title.

The assignment gives three example questions:
- How can I waive my annual fee?
- What should we offer to retain customer 15634602?
- Explain the difference between Gold and Platinum cards.

The first two may involve eligibility and account data not fully available in
the CSVs. No waiver or retention offer should be presented as approved.
The supplied documents contain specific conditions, which must be checked in
any reviewed test examples.

A final evaluation should record the model, retrieval settings, prompts,
expected policy passages, actual responses, and reviewer judgments for:
grounding accuracy, personalization, tone, and coherence. Do not substitute
synthetic unit-test pass rates for those qualitative measures. The supplied
private files and API credentials remain outside Git.
