import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models (both served locally via Ollama)
LLM_MODEL = "deepseek-r1:7b"
EMBED_MODEL = "nomic-embed-text"

# Internal knowledge base
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_DIR = os.path.join(BASE_DIR, "vectorstore")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Context fusion / top-K selection
TOP_K_VECTOR = 3
TOP_K_WEB = 2

# Evaluation
EVAL_LOG_PATH = os.path.join(BASE_DIR, "eval_logs", "runs.jsonl")
