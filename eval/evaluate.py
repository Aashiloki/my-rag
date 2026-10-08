"""
evaluate.py — Phase 3: Measure retrieval hit rate and refusal accuracy

WHAT THIS DOES:
Runs every question in questions.json through your retrieval pipeline
and measures two numbers:

1. RETRIEVAL HIT RATE
   For answerable questions: did the correct chunk appear in top-5?
   We check by looking for expected_keywords in the retrieved chunks.
   "Correct chunk found" = at least one retrieved chunk contains
   ALL the expected keywords for that question.

   Hit rate = (questions where correct chunk found) / (total answerable questions)
    The CI minimum is configured with --min-hit-rate.

2. REFUSAL ACCURACY
   For unanswerable questions: did the model correctly say
   "INSUFFICIENT_CONTEXT" instead of making something up?

   Refusal accuracy = (correct refusals) / (total unanswerable questions)
    The CI minimum is configured with --min-refusal-accuracy.

HOW TO RUN:
  python eval/evaluate.py

    # Enforce CI thresholds:
    python eval/evaluate.py --min-hit-rate 70 --min-refusal-accuracy 100

OUTPUT:
    Prints the measured rates and exits nonzero when a configured minimum is missed.
"""

import os
import sys
import json
import time
import argparse

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
REFUSAL_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "refusal_questions.json")


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        data = json.load(f)
    return data["questions"]


def load_refusal_questions() -> list[dict]:
    with open(REFUSAL_QUESTIONS_PATH) as f:
        data = json.load(f)
    return data["questions"]


def check_hit(retrieved_chunks: list[dict], expected_keywords: list[str]) -> bool:
    """
    Returns True if at least one retrieved chunk contains ALL expected keywords.

    LEARNING: This is a simple proxy for "did we retrieve the right chunk?"
    A production evaluation set would have exact chunk IDs, but keyword
    matching is a practical shortcut when building your first eval set.

    Lowercase comparison so "Membrane" matches "membrane".
    """
    if not expected_keywords:
        return False

    for chunk in retrieved_chunks:
        text_lower = chunk["text"].lower()
        if all(kw.lower() in text_lower for kw in expected_keywords):
            return True
    return False


def run_evaluation():
    from app.chat import ask
    from app.retrieval import retrieve

    answerable = load_questions()
    unanswerable = load_refusal_questions()
    questions = answerable + unanswerable

    if not answerable or not unanswerable:
        raise ValueError("Evaluation requires answerable and refusal questions.")

    print(f"\n{'='*60}")
    print(f"EVALUATION RUN")
    print(f"Total questions: {len(questions)} ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")
    print(f"{'='*60}\n")

    # ── Retrieval hit rate ─────────────────────────────────────────────────────
    hits = 0

    print("RETRIEVAL HIT RATE TEST")
    print("-" * 40)

    for q in answerable:
        chunks  = retrieve(q["question"])
        is_hit  = check_hit(chunks, q["expected_keywords"])
        hits   += int(is_hit)

        status = "✅ HIT " if is_hit else "❌ MISS"
        print(f"  Q{q['id']:02d} {status} | {q['question'][:55]}")

        time.sleep(0.1)  # small delay to avoid hammering the embedding model

    hit_rate = hits / len(answerable) * 100 if answerable else 0
    print(f"\nRetrieval hit rate: {hits}/{len(answerable)} = {hit_rate:.1f}%\n")

    # ── Refusal accuracy ───────────────────────────────────────────────────────
    correct_refusals = 0

    print("REFUSAL ACCURACY TEST")
    print("-" * 40)

    for q in unanswerable:
        chunks  = retrieve(q["question"])
        result  = ask(q["question"], chunks)
        refused = result["is_refusal"]

        correct_refusals += int(refused)
        status = "✅ REFUSED" if refused else "❌ ANSWERED (bad!)"
        print(f"  Q{q['id']:02d} {status} | {q['question'][:55]}")

        time.sleep(0.3)

    refusal_rate = correct_refusals / len(unanswerable) * 100 if unanswerable else 0
    print(f"\nRefusal accuracy: {correct_refusals}/{len(unanswerable)} = {refusal_rate:.1f}%")

    # ── Final summary ─────────────────────────────────────────────────────────
    chunk_size = os.environ.get("CHUNK_SIZE", "500")

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY:")
    print(f"{'='*60}")
    print(f"| chunk_size={chunk_size} | hit_rate={hit_rate:.1f}% | refusal={refusal_rate:.1f}% |")
    print(f"{'='*60}\n")

    return hit_rate, refusal_rate


def threshold_failures(
    hit_rate: float,
    refusal_rate: float,
    min_hit_rate: float | None,
    min_refusal_accuracy: float | None,
) -> list[str]:
    failures = []
    if min_hit_rate is not None and hit_rate < min_hit_rate:
        failures.append(f"hit rate {hit_rate:.1f}% is below {min_hit_rate:.1f}%")
    if min_refusal_accuracy is not None and refusal_rate < min_refusal_accuracy:
        failures.append(
            f"refusal accuracy {refusal_rate:.1f}% is below {min_refusal_accuracy:.1f}%"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and refusal quality.")
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-refusal-accuracy", type=float)
    args = parser.parse_args()

    for name, value in (
        ("--min-hit-rate", args.min_hit_rate),
        ("--min-refusal-accuracy", args.min_refusal_accuracy),
    ):
        if value is not None and not 0 <= value <= 100:
            parser.error(f"{name} must be between 0 and 100")

    hit_rate, refusal_rate = run_evaluation()
    failures = threshold_failures(
        hit_rate,
        refusal_rate,
        args.min_hit_rate,
        args.min_refusal_accuracy,
    )
    if failures:
        print("\nEvaluation gate failed: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
