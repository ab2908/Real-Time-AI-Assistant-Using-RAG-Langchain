# Real-Time-AI-Assistant-Using-RAG-Langchain
Developed a real-time AI assistant using a RAG framework with live web retrieval and open source LLMs to improve factual accuracy and response relevance.

A real-time **Retrieval-Augmented Generation (RAG)** AI assistant that combines the power of a local LLM ([LLaMA 3 8B](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/)) with live web search ([DuckDuckGo](https://duckduckgo.com/)) to answer your questions with up-to-date information.

---

## ✨ Features

- **Local LLM** — Runs LLaMA 3 (8B) locally through Ollama, keeping your data private.
- **Live Web Search** — Retrieves real-time search results from DuckDuckGo to ground the LLM's answers in current information.
- **RAG Pipeline** — Uses LangChain to chain search → prompt → LLM into a single, clean pipeline.
- **Interactive CLI** — Simple command-line chat interface; type a question, get an answer.

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
│  LLaMA 3 (8B)   │  ← Local LLM generates answer
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
