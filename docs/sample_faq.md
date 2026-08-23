# Frequently Asked Questions

## What is this assistant?

This is a real-time AI assistant built with a Retrieval-Augmented Generation
(RAG) pipeline. It answers questions by combining two sources of context:
your own internal documents (indexed with FAISS) and live web search
results (via DuckDuckGo), then feeds both to a locally-run LLM (any
Ollama-served chat model, configurable in `config.py`) through Ollama.

## How do I add my own documents?

Drop any `.txt`, `.md`, or `.pdf` files into the `docs/` folder, then run
`python ingest.py` to rebuild the FAISS index. Restart `assistant.py`
afterwards so it picks up the new index.

## Why does it also search the web?

Internal documents are great for private, stable knowledge (policies,
notes, specs) but go stale and can't cover current events. Web retrieval
fills that gap so answers stay grounded in up-to-date information.

## Is any data sent to the cloud?

No. The LLM and embedding model both run locally through Ollama. The only
outbound network call is the DuckDuckGo web search itself.
