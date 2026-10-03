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
   Target: start at ~61%, reach 89% after hybrid + re-ranking.

2. REFUSAL ACCURACY
   For unanswerable questions: did the model correctly say
   "INSUFFICIENT_CONTEXT" instead of making something up?

   Refusal accuracy = (correct refusals) / (total unanswerable questions)
   Target: 100% — every out-of-scope question should be refused.

HOW TO RUN:
  python eval/evaluate.py

  # Test a specific chunk size:
  CHUNK_SIZE=300 python eval/evaluate.py
  CHUNK_SIZE=800 python eval/evaluate.py

OUTPUT:
  Prints a table of results + final numbers to paste into your README.
"""

import os
import sys
import json
import time

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval import retrieve
from app.chat      import ask, build_prompt

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
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
    questions = load_questions()

    answerable   = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    print(f"\n{'='*60}")
    print(f"EVALUATION RUN")
    print(f"Total questions: {len(questions)} ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")
    print(f"{'='*60}\n")

    # ── Retrieval hit rate ─────────────────────────────────────────────────────
    hits        = 0
    hit_results = []

    print("RETRIEVAL HIT RATE TEST")
    print("-" * 40)

    for q in answerable:
        chunks  = retrieve(q["question"])
        is_hit  = check_hit(chunks, q["expected_keywords"])
        hits   += int(is_hit)

        status = "✅ HIT " if is_hit else "❌ MISS"
        hit_results.append({
            "id":       q["id"],
            "question": q["question"][:55],
            "hit":      is_hit,
        })
        print(f"  Q{q['id']:02d} {status} | {q['question'][:55]}")

        time.sleep(0.1)  # small delay to avoid hammering the embedding model

    hit_rate = hits / len(answerable) * 100 if answerable else 0
    print(f"\nRetrieval hit rate: {hits}/{len(answerable)} = {hit_rate:.1f}%\n")

    # ── Refusal accuracy ───────────────────────────────────────────────────────
    correct_refusals = 0
    refusal_results  = []

    print("REFUSAL ACCURACY TEST")
    print("-" * 40)

    for q in unanswerable:
        chunks  = retrieve(q["question"])
        result  = ask(q["question"], chunks)
        refused = result["is_refusal"]

        correct_refusals += int(refused)
        status = "✅ REFUSED" if refused else "❌ ANSWERED (bad!)"
        refusal_results.append({
            "id":      q["id"],
            "question": q["question"][:55],
            "refused": refused,
        })
        print(f"  Q{q['id']:02d} {status} | {q['question'][:55]}")

        time.sleep(0.3)

    refusal_rate = correct_refusals / len(unanswerable) * 100 if unanswerable else 0
    print(f"\nRefusal accuracy: {correct_refusals}/{len(unanswerable)} = {refusal_rate:.1f}%")

    # ── Final summary (paste into README) ─────────────────────────────────────
    chunk_size = os.environ.get("CHUNK_SIZE", "500")

    print(f"\n{'='*60}")
    print("README TABLE ROW (copy this):")
    print(f"{'='*60}")
    print(f"| chunk_size={chunk_size} | hit_rate={hit_rate:.1f}% | refusal={refusal_rate:.1f}% |")
    print(f"{'='*60}\n")

    return hit_rate, refusal_rate


if __name__ == "__main__":
    run_evaluation()
