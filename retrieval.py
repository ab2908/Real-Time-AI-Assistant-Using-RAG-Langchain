"""Vector search (FAISS), web retrieval (DuckDuckGo), and context fusion."""
import os

from ddgs import DDGS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config

_retriever = None
_retriever_loaded = False


def load_vector_retriever():
    """Loads the persisted FAISS index, if one exists. Returns None otherwise."""
    global _retriever, _retriever_loaded
    if _retriever_loaded:
        return _retriever

    _retriever_loaded = True
    if not os.path.isdir(config.INDEX_DIR):
        _retriever = None
        return None

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    index = FAISS.load_local(
        config.INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    _retriever = index.as_retriever(search_kwargs={"k": config.TOP_K_VECTOR})
    return _retriever


def vector_search(query):
    """Returns up to TOP_K_VECTOR Documents from the internal FAISS index."""
    retriever = load_vector_retriever()
    if retriever is None:
        return []
    return retriever.invoke(query)


def web_search(query):
    """Returns up to TOP_K_WEB Documents from a live DuckDuckGo search."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=config.TOP_K_WEB))

    return [
        Document(
            page_content=result.get("body", ""),
            metadata={
                "source": result.get("href", "unknown"),
                "title": result.get("title", "web result"),
            },
        )
        for result in results
    ]


def fuse_context(vector_docs, web_docs):
    """Merges the two retrieval branches into one numbered context string
    plus a matching list of human-readable source labels (Context Fusion +
    Top-K Relevant Context in the architecture diagram).
    """
    entries = [("internal", doc) for doc in vector_docs] + [
        ("web", doc) for doc in web_docs
    ]

    context_lines = []
    sources = []
    for i, (kind, doc) in enumerate(entries, start=1):
        label = doc.metadata.get("source", "unknown")
        context_lines.append(f"[{i}] ({kind}: {label})\n{doc.page_content}")
        sources.append({"index": i, "kind": kind, "source": label})

    context_text = "\n\n".join(context_lines) if context_lines else "(no context found)"
    return context_text, sources
