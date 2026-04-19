# Real-Time-AI-Assistant-Using-RAG-Langchain
Developed a real-time AI assistant using a RAG framework with live web retrieval and open source LLMs to improve factual accuracy and response relevance.

A real-time **Retrieval-Augmented Generation (RAG)** AI assistant that combines the power of a local LLM ([LLaMA 3 8B](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/)) with live web search ([DuckDuckGo](https://duckduckgo.com/)) to answer your questions with up-to-date information.

---

## ✨ Features

- **Local LLM** — Runs LLaMA 3 (8B) locally through Ollama, keeping your data private.
- **Live Web Search** — Retrieves real-time search results from DuckDuckGo to ground the LLM's answers in current information.
- **RAG Pipeline** — Uses LangChain to chain search → prompt → LLM into a single, clean pipeline.
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
| **English works best** | LLaMA 3 8B is strongest in English |
| **Speed depends on hardware** | Needs ~8 GB RAM; GPU speeds things up |
| **Search quality varies** | DuckDuckGo results may not always be perfectly relevant |

### 📊 Usage Limits

| Resource | Limit |
|---|---|
| **Ollama / LLaMA 3** | ♾️ Unlimited — local, no cost |
| **DuckDuckGo Search** | ~20–50 queries/min before throttling |
| **Disk Space** | ~4.7 GB for the model |
| **RAM** | ~8 GB while running |

> 💡 You can run it **24/7** with no subscription fees. The only constraint is DuckDuckGo's soft rate limit for rapid-fire queries.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  DuckDuckGo     │  ← Web search for relevant context
│  Search Tool    │
└────────┬────────┘
         │ search results
         ▼
┌─────────────────┐
│  Prompt Template│  ← Injects context + question
└────────┬────────┘
         │ formatted prompt
         ▼
┌─────────────────┐
│  LLaMA 3 (8B)  │  ← Local LLM generates answer
│  via Ollama     │
└────────┬────────┘
         │
         ▼
      Answer
```

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or higher |
| **Ollama** | Installed and running ([download](https://ollama.com/download)) |
| **LLaMA 3 model** | Pulled via Ollama (see setup below) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Rag Ai assistant"
```

### 2. Install Python dependencies

```bash
pip install langchain-ollama langchain-community langchain-core duckduckgo-search
```

### 3. Install & start Ollama

Download Ollama from [ollama.com/download](https://ollama.com/download) and install it for your OS.

### 4. Pull the LLaMA 3 model

```bash
ollama pull llama3:8b
```

> [!NOTE]
> The model is approximately **4.7 GB**. Make sure you have enough disk space and a stable internet connection.

### 5. Run the assistant

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

You: What are the symptoms of vitamin D deficiency?
🤖 Thinking...
🤖: According to the search results, common symptoms include fatigue,
    bone pain, muscle weakness...

You: exit
🤖 Goodbye!
```

---

## 📁 Project Structure

```
Rag Ai assistant/
├── assistant.py    # Main application — RAG chain & chat loop
└── README.md       # This file
```

---

## ⚙️ Configuration

| Setting | Location | Default | Description |
|---|---|---|---|
| **LLM Model** | `assistant.py` line 7 | `llama3:8b` | Change to any Ollama-supported model |
| **Search Tool** | `assistant.py` line 10 | DuckDuckGo | Swap with another LangChain search tool |

To use a different model, update this line in `assistant.py`:

```python
llm = OllamaLLM(model="llama3:8b")  # ← change model name here
```

---

## 🛠️ Tech Stack

- [**LangChain**](https://www.langchain.com/) — Orchestration framework for the RAG pipeline
- [**Ollama**](https://ollama.com/) — Local LLM runtime
- [**LLaMA 3**](https://ollama.com/library/llama3) — Meta's open-source large language model
- [**DuckDuckGo Search**](https://pypi.org/project/duckduckgo-search/) — Privacy-focused web search API

---



## 📄 License

This project is open-source. Feel free to use, modify, and distribute.
