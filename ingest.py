"""Builds (or rebuilds) the FAISS index over the internal documents in docs/.

Run this once before the first `python assistant.py`, and again any time
files in docs/ are added, removed, or edited.
"""
import sys

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

_LOADERS = {
    "**/*.md": (TextLoader, {"encoding": "utf-8"}),
    "**/*.txt": (TextLoader, {"encoding": "utf-8"}),
    "**/*.pdf": (PyPDFLoader, {}),
}


def load_documents():
    documents = []
    for pattern, (loader_cls, loader_kwargs) in _LOADERS.items():
        loader = DirectoryLoader(
            config.DOCS_DIR,
            glob=pattern,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
        )
        documents.extend(loader.load())
    return documents


def main():
    print(f"Loading documents from {config.DOCS_DIR} ...")
    documents = load_documents()

    if not documents:
        print(
            "No .md, .txt, or .pdf files found in docs/. Add some documents "
            "and re-run this script."
        )
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} document(s) -> {len(chunks)} chunk(s).")

    print(f"Embedding chunks with '{config.EMBED_MODEL}' via Ollama ...")
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    index = FAISS.from_documents(chunks, embeddings)

    index.save_local(config.INDEX_DIR)
    print(f"Saved FAISS index to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
