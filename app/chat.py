"""
chat.py — Phase 3: Citations, refusal, token tracking, query cache

NEW IN PHASE 3:
1. Inline citations — [Source N] after each claim
2. Explicit refusal prompt — tested against unanswerable questions
3. Token cost tracking — prompt + completion tokens logged per query
4. Simple in-memory cache — skip embedding + LLM for repeated questions
"""

import os
import hashlib
from groq import Groq

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# The legacy Llama names used in older tutorials are no longer available for
# all accounts. Use a currently accessible Groq model instead.
MODEL        = "qwen/qwen3.8-27b"
MAX_TOKENS   = 1024

# ─── CACHE ─────────────────────────────────────────────────────────────────────
# LEARNING: A cache stores the result of expensive operations so repeated
# calls return instantly. Here: if the same question is asked twice,
# we skip the Qdrant search + Groq API call (saves ~1-2 seconds + API tokens).
#
# This is an in-memory cache — it resets when the server restarts.
# Phase 4 would use Redis for a persistent cache across restarts.
#
# Key: MD5 hash of the question string
# Value: the full answer string
_cache: dict[str, str] = {}

# ─── TOKEN TRACKING ────────────────────────────────────────────────────────────
# Simple accumulator — resets on server restart.
# Phase 4 would persist this to a database.
token_stats = {
    "total_queries":           0,
    "total_prompt_tokens":     0,
    "total_completion_tokens": 0,
    "cache_hits":              0,
}


def clear_cache() -> None:
    """Drop the in-memory answer cache so repeated questions do not reuse stale data after re-ingestion."""
    _cache.clear()
    print("[chat] Cache cleared after document refresh.")


def get_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "PowerShell: $env:GROQ_API_KEY='gsk_your_key'\n"
            "Then restart uvicorn."
        )
    return Groq(api_key=GROQ_API_KEY)


# ─── PROMPT BUILDER ────────────────────────────────────────────────────────────
def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Phase 3 prompt — three upgrades over Phase 2:

    1. Numbered sources so the LLM can cite them inline as [Source N]
    2. Explicit "cite your sources" instruction
    3. Explicit "refuse if not in context" instruction with exact wording
       the evaluate.py script checks for

    LEARNING: Prompt instructions must be specific to work reliably.
    "Be helpful" → vague, model ignores it under pressure.
    "If context doesn't contain the answer, output exactly:
     INSUFFICIENT_CONTEXT" → specific, testable, measurable.
    We use that exact token in evaluate.py to detect refusals.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}"
        )
    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""You are a precise research assistant. Answer questions using ONLY the sources provided below.

RULES — follow all of them:
1. Use ONLY the provided sources. Do not use outside knowledge.
2. After each factual claim, cite the source as [Source N].
3. If the sources do not contain enough information to answer, output exactly:
   INSUFFICIENT_CONTEXT
4. Do not speculate or infer beyond what the sources explicitly state.
5. Be concise. Do not repeat the question.

SOURCES:
{context_str}

QUESTION:
{question}

ANSWER:"""


# ─── MAIN ASK FUNCTION ─────────────────────────────────────────────────────────
def ask(question: str, chunks: list[dict]) -> dict:
    """
    Returns a dict with:
    - answer:            str  (the LLM response, or refusal text)
    - is_refusal:        bool (True if model said INSUFFICIENT_CONTEXT)
    - cache_hit:         bool
    - prompt_tokens:     int
    - completion_tokens: int

    LEARNING: returning a dict instead of just a string lets main.py
    expose token counts and cache status in the API response without
    needing global variables or extra function calls.
    """
    token_stats["total_queries"] += 1

    if not chunks:
        return {
            "answer":            "No relevant documents found. Please rephrase your question.",
            "is_refusal":        True,
            "cache_hit":         False,
            "prompt_tokens":     0,
            "completion_tokens": 0,
        }

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = hashlib.md5(question.strip().lower().encode()).hexdigest()
    if cache_key in _cache:
        token_stats["cache_hits"] += 1
        print(f"[chat] Cache hit for: '{question[:60]}'")
        return {
            "answer":            _cache[cache_key],
            "is_refusal":        "INSUFFICIENT_CONTEXT" in _cache[cache_key],
            "cache_hit":         True,
            "prompt_tokens":     0,
            "completion_tokens": 0,
        }

    # ── LLM call ──────────────────────────────────────────────────────────────
    groq_client = get_client()
    prompt      = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = MAX_TOKENS,
        temperature = 0.1,
    )

    raw_answer = response.choices[0].message.content.strip()
    usage      = response.usage

    # ── Token tracking ────────────────────────────────────────────────────────
    token_stats["total_prompt_tokens"]     += usage.prompt_tokens
    token_stats["total_completion_tokens"] += usage.completion_tokens

    print(
        f"[chat] Tokens — prompt:{usage.prompt_tokens} "
        f"completion:{usage.completion_tokens} "
        f"total:{usage.total_tokens}"
    )

    # ── Detect refusal ────────────────────────────────────────────────────────
    # LEARNING: we told the model to output "INSUFFICIENT_CONTEXT" as a
    # machine-readable signal. evaluate.py checks for this exact string
    # to measure refusal accuracy on unanswerable test questions.
    is_refusal = "INSUFFICIENT_CONTEXT" in raw_answer
    if is_refusal:
        answer = "I don't have enough information in my documents to answer this question."
    else:
        answer = raw_answer

    # ── Cache the result ──────────────────────────────────────────────────────
    _cache[cache_key] = raw_answer   # store raw so refusal detection works on re-read

    return {
        "answer":            answer,
        "is_refusal":        is_refusal,
        "cache_hit":         False,
        "prompt_tokens":     usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }
