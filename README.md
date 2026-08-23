# Real-Time-AI-Assistant-Using-RAG-Langchain
Developed a real-time AI assistant using a hybrid RAG framework with FAISS-indexed internal documents, live web retrieval, and open source LLMs to improve factual accuracy and response relevance.

A **Retrieval-Augmented Generation (RAG)** AI assistant that fuses two sources of context — a local **FAISS** vector index over your own documents, and live **DuckDuckGo** web search — before asking a local LLM (any [Ollama](https://ollama.com/)-served chat model — defaults to **DeepSeek-R1 7B**) to produce a grounded, sourced answer. Every run is scored and logged for latency and groundedness.

---

## ✨ Features

- **Local LLM** — Runs an open-source chat model (DeepSeek-R1, Gemma 3, Llama 3, Qwen, etc.) locally through Ollama, keeping your data private.
- **Hybrid Retrieval** — Combines a FAISS vector search over your own indexed documents with live DuckDuckGo web search, run in parallel.
- **Context Fusion** — Merges and caps both retrieval branches into a single top-K context block before it ever reaches the LLM.
- **Sourced Answers** — Every answer is printed alongside the specific internal documents / web pages it was grounded in.
- **Evaluation** — Each turn logs per-stage latency plus an LLM-judged groundedness score to `eval_logs/runs.jsonl`.
- **Interactive CLI** — Simple command-line chat interface; type a question, get an answer.
- **100% Free & Unlimited** — No API keys, no subscriptions. Runs entirely on your machine.

---

## 🎯 What Can You Use It For?

### ✅ Great For

| Use Case | Example Question |
|---|---|
| 📰 **Current events & news** | "What happened in tech today?" |
| 🔍 **Fact-checking** | "Is it true that X happened?" |
| 📚 **Research & learning** | "What are the side effects of vitamin D?" |
| 💡 **Explanations** | "Explain quantum computing in simple terms" |
| ⚖️ **Comparisons** | "Compare React vs Angular in 2026" |
| 🛠️ **How-to guides** | "How to apply for a US visa?" |
| 🛒 **Product lookups** | "What is the latest iPhone model?" |
| 📊 **Market & trends** | "What are trending programming languages?" |
| 🌍 **Travel info** | "Best time to visit Japan?" |
| 🏥 **Health questions** | "What are symptoms of dehydration?" |

> Essentially — **any question that can be answered by a web search + AI reasoning**.

### ⚠️ Limitations

| Limitation | Details |
|---|---|
| **No memory** | Each question is independent; it doesn't remember previous chat |
| **No file upload** | It only searches the web, not your local files |
| **Text only** | No image, audio, or video input/output |
| **English works best** | Most local models are strongest in English |
| **Speed depends on hardware** | Needs ~4-8 GB RAM depending on the model; GPU speeds things up |
| **Search quality varies** | DuckDuckGo results may not always be perfectly relevant |

### 📊 Usage Limits

| Resource | Limit |
|---|---|
| **Ollama / local LLM** | ♾️ Unlimited — local, no cost |
| **DuckDuckGo Search** | ~20–50 queries/min before throttling |
| **Disk Space** | ~3-5 GB for the chat model + ~275 MB for the embedding model |
| **RAM** | ~4-8 GB while running, depending on the model |

> 💡 You can run it **24/7** with no subscription fees. The only constraint is DuckDuckGo's soft rate limit for rapid-fire queries.

---

## 🏗️ Architecture

```
                      USER
                           │
                           ▼
                    ┌──────────────┐
                    │ User Query   │
                    └──────┬───────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Query Processing │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐
     │ Vector Search   │       │  Web Retrieval  │
     │     FAISS       │       │   DuckDuckGo    │
     │                 │       │                 │
     │ Internal Docs   │       │ Current Data    │
     └────────┬────────┘       └────────┬────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                 ┌──────────────────┐
                 │ Context Fusion   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Top-K Relevant   │
                 │ Context          │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Prompt Template  │
                 │ Query + Context  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Chat LLM         │
                 │ via Ollama        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Grounded Answer  │
                 │ + Sources        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Evaluation       │
                 │ Accuracy/Latency │
                 └──────────────────┘
```

The two retrieval branches (FAISS over `docs/`, and DuckDuckGo web search) run **in parallel**. `retrieval.py` implements both branches plus `fuse_context()` (Context Fusion + Top-K selection); `pipeline.py` wires the whole thing together and times each stage; `evaluation.py` scores groundedness and appends a JSON record per turn to `eval_logs/runs.jsonl`.

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or higher |
| **Ollama** | Installed and running ([download](https://ollama.com/download)) |
| **A chat model** | Any Ollama-served chat model — default is `deepseek-r1:7b` (see setup below) |
| **nomic-embed-text model** | Pulled via Ollama, used for FAISS embeddings (see setup below) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Rag Ai assistant"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install & start Ollama

Download Ollama from [ollama.com/download](https://ollama.com/download) and install it for your OS.

### 4. Pull the LLM and embedding models

```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

> [!NOTE]
> Already have other Ollama models pulled (Llama 3, Gemma 3, Qwen, etc.)? You can point `LLM_MODEL` in `config.py` at any of them instead — see [Configuration](#️-configuration) below. `nomic-embed-text` (~275 MB) is still required either way, since it's the embedding model FAISS uses; a chat model can't substitute for it. `deepseek-r1` is a reasoning model that internally produces `<think>...</think>` output before its final answer — in testing this integrated cleanly with the groundedness/correctness scoring in `evaluation.py`, but if you swap in a different reasoning model and see scores that look off, check whether raw `<think>` text is leaking into the parsed response.

### 5. Build the FAISS index over your internal documents

Drop `.md` / `.txt` / `.pdf` files into `docs/` (a couple of markdown samples are included so this works out of the box), then run:

```bash
python ingest.py
```

Re-run this any time you add, remove, or edit files in `docs/`.

### 6. Run the assistant

```bash
python assistant.py
```

You should see:

```
🤖 Hello! I'm a real-time AI assistant. What's new?
You: 
```

Type any question and press **Enter**. Type `exit` or `quit` to stop.

---

## 💬 Usage Example

```
🤖 Hello! I'm a real-time AI assistant. What's new?
You: What is the latest news about AI?
🤖 Thinking...
🤖: Based on the search results, here are the latest developments in AI...

📚 Sources:
  [1] internal: docs/sample_notes.md
  [2] web: https://example.com/some-article

⏱️  vector=42.1ms | web=610.4ms (parallel) | llm=2950.7ms | total=3605.9ms
📊 groundedness: 4/5

You: exit
🤖 Goodbye!
```

---

## 📁 Project Structure

```
Rag Ai assistant/
├── assistant.py     # CLI entry point — chat loop, prints answer/sources/eval
├── config.py         # Models, paths, chunk size, top-K settings
├── ingest.py         # Builds/rebuilds the FAISS index from docs/
├── retrieval.py       # Vector search (FAISS), web search (DuckDuckGo), context fusion
├── pipeline.py        # Orchestrates query processing → retrieval → prompt → LLM
├── evaluation.py       # Latency logging + LLM-judged groundedness/correctness scoring
├── benchmark.py        # Runs eval_set.json through the pipeline, reports accuracy/precision/recall/F1
├── eval_set.json       # Labeled test questions (expected source doc + reference answer)
├── docs/              # Internal documents indexed by ingest.py
├── vectorstore/        # Persisted FAISS index (generated, gitignored)
├── eval_logs/          # Per-run JSONL evaluation log (generated, gitignored)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 📏 Benchmarking

`eval_set.json` holds a small labeled test set — questions with a known
correct answer and known source document. Run it end to end with:

```bash
python benchmark.py
```

This reports:

- **Answer accuracy** — % of answers an LLM judge rates as matching the reference answer
- **Retrieval precision / recall / F1** — whether FAISS retrieved the actual correct source document for each question
- **Groundedness** — the same 1–5 per-answer score used in normal chat, averaged
- **Latency** — average end-to-end response time

Full per-question results are saved to `eval_logs/benchmark_results.json`. These are real measured numbers from your own run, not fixed constants — expect them to vary by hardware and by which local model you're running.

---

## ⚙️ Configuration

All tunable settings live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| **LLM Model** | `deepseek-r1:7b` | Change to any Ollama-supported chat model already pulled on your machine |
| **Embedding Model** | `nomic-embed-text` | Change to any Ollama-supported embedding model |
| **CHUNK_SIZE / CHUNK_OVERLAP** | `500` / `50` | Document splitting for the FAISS index |
| **TOP_K_VECTOR / TOP_K_WEB** | `3` / `2` | Max results pulled from each retrieval branch before fusion |

To use a different model, update `config.py`:

```python
LLM_MODEL = "deepseek-r1:7b"      # ← change chat model here (e.g. any `ollama list` entry)
EMBED_MODEL = "nomic-embed-text"  # ← change embedding model here
```

If you change `EMBED_MODEL`, re-run `python ingest.py` to rebuild the FAISS index with the new embeddings.

---

## 🛠️ Tech Stack

- [**LangChain**](https://www.langchain.com/) — Orchestration framework for the RAG pipeline
- [**Ollama**](https://ollama.com/) — Local LLM & embedding runtime
- [**DeepSeek-R1**](https://ollama.com/library/deepseek-r1) — Open-source reasoning chat model (default `LLM_MODEL`; swappable for any Ollama chat model)
- [**nomic-embed-text**](https://ollama.com/library/nomic-embed-text) — Local embedding model for the FAISS index
- [**FAISS**](https://github.com/facebookresearch/faiss) — Vector similarity search over internal documents
- [**ddgs**](https://pypi.org/project/ddgs/) — Privacy-focused DuckDuckGo web search API

---



## 📄 License

This project is open-source. Feel free to use, modify, and distribute.
