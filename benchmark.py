"""Runs the labeled questions in eval_set.json through the full pipeline
and reports answer accuracy, retrieval precision/recall/F1, groundedness,
and latency -- the numbers to cite when asked "what did you see when you
benchmarked it".

Requires a running Ollama server with both models pulled, and a FAISS
index already built (run ingest.py first).
"""
import json
import os
import statistics

import evaluation
import pipeline

EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_logs", "benchmark_results.json"
)


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(item):
    result = pipeline.run(item["question"])

    retrieved_internal = [s["source"] for s in result.sources if s["kind"] == "internal"]
    retrieval = evaluation.retrieval_metrics(retrieved_internal, [item["expected_source"]])

    correct = evaluation.score_correctness(
        pipeline.llm, item["question"], result.answer, item["reference_answer"]
    )

    return {
        "question": item["question"],
        "expected_source": item["expected_source"],
        "answer": result.answer,
        "correct": correct,
        "retrieval": retrieval,
        "groundedness": result.groundedness,
        "latencies_ms": result.latencies_ms,
    }


def summarize(results):
    judged = [r for r in results if r["correct"] is not None]
    accuracy = sum(1 for r in judged if r["correct"]) / len(judged) if judged else None

    precision = statistics.mean(r["retrieval"]["precision"] for r in results)
    recall = statistics.mean(r["retrieval"]["recall"] for r in results)
    f1 = statistics.mean(r["retrieval"]["f1"] for r in results)

    grounded = [r["groundedness"] for r in results if r["groundedness"] is not None]
    avg_groundedness = statistics.mean(grounded) if grounded else None

    avg_latency = statistics.mean(r["latencies_ms"]["total"] for r in results)

    return {
        "questions": len(results),
        "accuracy": accuracy,
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_f1": f1,
        "avg_groundedness": avg_groundedness,
        "avg_latency_ms": avg_latency,
    }


def print_summary(summary):
    print("\n=== Benchmark summary ===")
    print(f"Questions:            {summary['questions']}")
    if summary["accuracy"] is not None:
        print(f"Answer accuracy:      {summary['accuracy'] * 100:.1f}%")
    print(f"Retrieval precision:  {summary['retrieval_precision'] * 100:.1f}%")
    print(f"Retrieval recall:     {summary['retrieval_recall'] * 100:.1f}%")
    print(f"Retrieval F1:         {summary['retrieval_f1'] * 100:.1f}%")
    if summary["avg_groundedness"] is not None:
        print(f"Avg groundedness:     {summary['avg_groundedness']:.2f}/5")
    print(f"Avg total latency:    {summary['avg_latency_ms']:.0f}ms")


def main():
    eval_set = load_eval_set()
    results = []

    for i, item in enumerate(eval_set, start=1):
        print(f"[{i}/{len(eval_set)}] {item['question']}")
        results.append(run_one(item))

    summary = summarize(results)
    print_summary(summary)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
