"""Evaluation stage: latency logging + an LLM-judged groundedness score
(a proxy for accuracy, since there is no labeled Q&A test set)."""
import json
import os
import re
from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate

import config

_groundedness_prompt = ChatPromptTemplate.from_template(
    """Rate how well the ANSWER is supported *only* by the CONTEXT below,
    on a scale of 1 (not supported at all) to 5 (fully supported).
    Reply with just the single digit and nothing else.

    CONTEXT:
    {context}

    ANSWER:
    {answer}
    """
)

_correctness_prompt = ChatPromptTemplate.from_template(
    """You are grading a question-answering system. Given a REFERENCE ANSWER
    (assumed correct) and a GENERATED ANSWER, decide whether the generated
    answer conveys the same key facts as the reference answer, even if
    worded differently. Reply with exactly one word: CORRECT or INCORRECT.

    QUESTION:
    {question}

    REFERENCE ANSWER:
    {reference_answer}

    GENERATED ANSWER:
    {generated_answer}
    """
)


def score_groundedness(llm, answer, context):
    """Returns an int 1-5, or None if the judge call fails or can't be parsed."""
    try:
        raw = (_groundedness_prompt | llm).invoke(
            {"context": context, "answer": answer}
        )
        match = re.search(r"[1-5]", raw)
        return int(match.group()) if match else None
    except Exception:
        return None


def score_correctness(llm, question, generated_answer, reference_answer):
    """LLM-judged correctness against a labeled reference answer.

    Returns True/False, or None if the judge call fails or can't be parsed.
    """
    try:
        raw = (_correctness_prompt | llm).invoke(
            {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
            }
        )
        verdict = raw.strip().upper()
        if "INCORRECT" in verdict:
            return False
        if "CORRECT" in verdict:
            return True
        return None
    except Exception:
        return None


def retrieval_metrics(retrieved_sources, expected_sources):
    """Precision/recall/F1 of the internal-doc retrieval branch for one
    query, comparing retrieved document filenames against the known
    relevant filename(s) for that question."""
    retrieved = {os.path.basename(s) for s in retrieved_sources}
    expected = set(expected_sources)

    true_positives = len(retrieved & expected)
    false_positives = len(retrieved - expected)
    false_negatives = len(expected - retrieved)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def log_run(record):
    """Appends one JSON line describing this run to EVAL_LOG_PATH."""
    os.makedirs(os.path.dirname(config.EVAL_LOG_PATH), exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(config.EVAL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
