import sys

# Windows consoles/pipes often default to a non-UTF-8 codepage, which
# crashes on the emoji used below. Force UTF-8 output so this runs the
# same in Windows Terminal, PowerShell, redirected output, etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
import pipeline
from retrieval import load_vector_retriever


def print_result(result):
    print(f"🤖: {result.answer}")

    if result.sources:
        print("\n📚 Sources:")
        for s in result.sources:
            print(f"  [{s['index']}] {s['kind']}: {s['source']}")

    lat = result.latencies_ms
    groundedness = f"{result.groundedness}/5" if result.groundedness is not None else "n/a"
    print(
        f"\n⏱️  vector={lat['vector_search']}ms | web={lat['web_search']}ms "
        f"(parallel) | llm={lat['llm']}ms | total={lat['total']}ms"
    )
    print(f"📊 groundedness: {groundedness}")


def main():
    if load_vector_retriever() is None:
        print(
            "⚠️  No FAISS index found in "
            f"{config.INDEX_DIR}. Run `python ingest.py` to index the "
            "documents in docs/. Continuing with web search only for now."
        )

    print("🤖 Hello! I'm a real-time AI assistant. What's new?")
    while True:
        try:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                print("🤖 Goodbye!")
                break

            print("🤖 Thinking...")
            result = pipeline.run(user_query)
            print_result(result)

        except ValueError as e:
            print(f"⚠️  {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
