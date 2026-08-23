"""Orchestrates the full RAG pipeline: query processing -> parallel
retrieval -> context fusion -> prompt -> LLM -> grounded answer -> eval.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

import config
import evaluation
from retrieval import fuse_context, vector_search, web_search

llm = OllamaLLM(model=config.LLM_MODEL)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. Answer the user's question based
    *only* on the following context. If the context is empty or does not
    contain the answer, say 'I could not find any information on that.'

    Context:
    {context}

    Question:
    {question}
    """
)


@dataclass
class AnswerResult:
    answer: str
    sources: list
    latencies_ms: dict
    groundedness: int = None
    error: str = field(default=None)


def process_query(raw_query):
    """Query Processing stage: normalize whitespace, reject empty input."""
    query = " ".join(raw_query.split())
    if not query:
        raise ValueError("Query is empty.")
    return query


def run(raw_query):
    total_start = time.perf_counter()
    query = process_query(raw_query)

    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_start = time.perf_counter()
        web_start = time.perf_counter()
        vector_future = executor.submit(vector_search, query)
        web_future = executor.submit(web_search, query)

        vector_docs = vector_future.result()
        vector_latency_ms = (time.perf_counter() - vector_start) * 1000

        web_docs = web_future.result()
        web_latency_ms = (time.perf_counter() - web_start) * 1000

    context, sources = fuse_context(vector_docs, web_docs)

    llm_start = time.perf_counter()
    answer = (prompt | llm).invoke({"context": context, "question": query})
    llm_latency_ms = (time.perf_counter() - llm_start) * 1000

    groundedness = evaluation.score_groundedness(llm, answer, context)

    total_latency_ms = (time.perf_counter() - total_start) * 1000
    latencies_ms = {
        "vector_search": round(vector_latency_ms, 1),
        "web_search": round(web_latency_ms, 1),
        "llm": round(llm_latency_ms, 1),
        "total": round(total_latency_ms, 1),
    }

    evaluation.log_run(
        {
            "query": query,
            "vector_hits": len(vector_docs),
            "web_hits": len(web_docs),
            "latencies_ms": latencies_ms,
            "groundedness": groundedness,
        }
    )

    return AnswerResult(
        answer=answer,
        sources=sources,
        latencies_ms=latencies_ms,
        groundedness=groundedness,
    )
