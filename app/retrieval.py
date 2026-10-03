"""
retrieval.py — Phase 3: Hybrid search + cross-encoder re-ranking

NEW IN PHASE 3:
1. Hybrid retrieval: BM25 keyword score + vector similarity score combined
2. Cross-encoder re-ranking: a second, slower model re-scores the top candidates

PHASE 2 vs PHASE 3 RETRIEVAL:

Phase 2:
  query → embed → Qdrant top-5 → done

Phase 3:
  query → embed → Qdrant top-20  (vector)
         ↘ tokenise → BM25 top-20 (keyword)
            → merge + RRF score   (hybrid)
            → cross-encoder       (re-rank)
            → top-5 returned

WHY THIS ORDER?
- We fetch 20 from each source (not 5) because some relevant chunks
  score poorly in one modality but well in the other.
- Merging gives us a richer candidate pool.
- The cross-encoder then re-scores that pool more accurately than
  either bi-encoder (MiniLM) or BM25 alone.
- Final top-5 comes from the cross-encoder scores.

WHAT IS A CROSS-ENCODER?
The embedding model (bi-encoder) encodes query and chunk independently,
then compares vectors. Fast, but loses the interaction between words.

A cross-encoder sees BOTH query and chunk at once:
  input:  "[query] What is osmosis? [SEP] Osmosis is the movement..."
  output: a single relevance score 0→1

Much more accurate, but too slow to run on thousands of chunks.
So: bi-encoder/BM25 for fast candidate selection → cross-encoder for
accurate final ranking. This two-stage approach is industry standard.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.db import client, COLLECTION_NAME

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BM25_PATH       = os.path.join(os.path.dirname(__file__), "..", "bm25_index.pkl")
CANDIDATE_K     = 20    # fetch this many from each source before re-ranking
FINAL_K         = 5     # return this many after re-ranking
RRF_K           = 60    # RRF constant — controls how much rank matters vs score

# ─── LAZY MODEL LOADING ───────────────────────────────────────────────────────
# Railway containers can crash or time out if we try to download/initialize
# these heavy models at import time. We only create them when a request
# actually needs them.
_bi_encoder = None
_cross_encoder = None


def _get_bi_encoder():
    global _bi_encoder
    if _bi_encoder is None:
        print("[retrieval] Loading bi-encoder (embedding model)...")
        _bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _bi_encoder


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("[retrieval] Loading cross-encoder (re-ranker)...")
        # ms-marco-MiniLM-L-6-v2: trained specifically for passage re-ranking.
        # Downloads ~80 MB once, then cached. Free, runs locally.
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ─── LOAD BM25 INDEX ───────────────────────────────────────────────────────────
def _load_bm25():
    if not os.path.exists(BM25_PATH):
        return None, None, None
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"], data["payloads"]


# ─── RECIPROCAL RANK FUSION ────────────────────────────────────────────────────
def _rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """
    Merges multiple ranked lists into one score per item.

    LEARNING: RRF is the standard way to combine two ranked lists
    without needing to normalise their scores (BM25 and cosine scores
    live on completely different scales — you can't just add them).

    Formula: score(item) = sum over each ranking of  1 / (k + rank)
    where rank is 1-indexed position in that list.

    k=60 is the standard default (from the original RRF paper).
    Higher k = rank differences matter less. Lower k = top rank matters more.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


# ─── MAIN RETRIEVAL FUNCTION ───────────────────────────────────────────────────
def retrieve(query: str, top_k: int = FINAL_K) -> list[dict]:
    """
    Full Phase 3 retrieval pipeline.
    Returns top_k chunks, each with text, source, chunk_index, score.
    """
    if client is None:
        print("[retrieval] ⚠️  Qdrant is unavailable; skip retrieval until the DB is available.")
        return []

    # ── 1. Vector retrieval via Qdrant ────────────────────────────────────────
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print("[retrieval] ❌  Run ingest first.")
        return []

    bi_encoder = _get_bi_encoder()
    query_vector = bi_encoder.encode(query).tolist()
    vector_results = client.search(
        collection_name = COLLECTION_NAME,
        query_vector    = query_vector,
        limit           = CANDIDATE_K,
        with_payload    = True,
    )

    # Map chunk_id → payload for later lookup
    id_to_payload: dict[str, dict] = {}
    vector_ranking: list[str]      = []

    for point in vector_results:
        cid = str(point.id)
        id_to_payload[cid] = point.payload or {}
        vector_ranking.append(cid)

    # ── 2. BM25 keyword retrieval ─────────────────────────────────────────────
    bm25, all_chunks, all_payloads = _load_bm25()
    bm25_ranking: list[str] = []

    if bm25 is not None:
        tokenised_query = query.lower().split()
        bm25_scores     = bm25.get_scores(tokenised_query)
        # Get top CANDIDATE_K indices sorted by score descending
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]

        for idx in top_bm25_indices:
            cid = str(idx)
            bm25_ranking.append(cid)
            if cid not in id_to_payload:
                id_to_payload[cid] = all_payloads[idx]
    else:
        print("[retrieval] ⚠  BM25 index not found — using vector only")

    # ── 3. Merge with Reciprocal Rank Fusion ─────────────────────────────────
    rankings  = [vector_ranking, bm25_ranking] if bm25_ranking else [vector_ranking]
    rrf_scores = _rrf(rankings)

    # Take top CANDIDATE_K after RRF merge
    merged_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:CANDIDATE_K]

    # ── 4. Cross-encoder re-ranking ───────────────────────────────────────────
    # LEARNING: CrossEncoder.predict() takes a list of [query, passage] pairs.
    # It processes them all in one batched forward pass — efficient.
    cross_encoder = _get_cross_encoder()
    candidates = [id_to_payload[cid].get("text", "") for cid in merged_ids]
    pairs = [[query, text] for text in candidates]
    ce_scores = cross_encoder.predict(pairs)

    # Sort by cross-encoder score (higher = more relevant)
    reranked = sorted(
        zip(merged_ids, ce_scores),
        key    = lambda x: x[1],
        reverse= True,
    )[:top_k]

    # ── 5. Build return list ──────────────────────────────────────────────────
    results = []
    for cid, ce_score in reranked:
        payload = id_to_payload.get(cid, {})
        results.append({
            "text":        payload.get("text", ""),
            "source":      payload.get("source", "unknown"),
            "chunk_index": payload.get("chunk_index", -1),
            "score":       round(float(ce_score), 4),
        })

    return results


# ─── DEBUG ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = "What is osmosis?"
    print(f"\n[retrieval] Query: '{q}'\n")
    for i, r in enumerate(retrieve(q)):
        print(f"--- Chunk {i+1} (score={r['score']}) ---")
        print(f"Source: {r['source']} | Chunk #{r['chunk_index']}")
        print(r["text"][:300])
        print()
