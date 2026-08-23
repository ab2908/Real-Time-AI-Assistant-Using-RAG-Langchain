# Project Notes

## Architecture summary

Query Processing -> (Vector Search over internal docs via FAISS) and
(Web Retrieval via DuckDuckGo) run in parallel -> Context Fusion ->
Top-K Relevant Context -> Prompt Template -> Llama 3 8B via Ollama ->
Grounded Answer + Sources -> Evaluation (latency + groundedness).

## Evaluation approach

Since there is no labeled question/answer test set, "accuracy" is
approximated with an LLM-judged groundedness score: after generating an
answer, the same local LLM is asked to rate, from 1 to 5, how well the
answer is supported *only* by the retrieved context. This is logged
alongside per-stage latency (vector search, web search, LLM generation,
total) to `eval_logs/runs.jsonl` for later analysis.

## Known limitations

- Groundedness scoring is a heuristic, not ground truth.
- DuckDuckGo search quality and rate limits vary.
- The FAISS index must be rebuilt manually (`python ingest.py`) after
  documents in `docs/` change; it does not watch the folder.
