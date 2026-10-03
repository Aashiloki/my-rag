"""
ingest.py — Phase 3: Ingestion with BM25 index + Qdrant vectors

NEW IN PHASE 3:
- Builds a BM25 index (keyword search) alongside Qdrant (vector search)
- Saves BM25 index to disk so retrieval.py can load it
- Chunk size is now configurable via CHUNK_SIZE env var so you can
  experiment and re-measure without editing code

WHAT IS BM25?
BM25 is the algorithm behind classic keyword search (used by Elasticsearch,
Lucene, Google's early search). It scores a document by how often query
words appear in it, weighted by how rare those words are across all docs.
- "the" appears everywhere → low weight
- "mitochondria" appears rarely → high weight
BM25 is fast, needs no GPU, and is excellent at exact-term matching.

WHY BOTH BM25 AND VECTOR?
Vector search: great at semantic meaning, misses exact terms
BM25 search:   great at exact terms, misses paraphrases
Hybrid:        catches both cases → significantly higher hit rate
This is what the spec calls "hybrid retrieval" and it's a key Phase 3 feature.
"""

import os
import pickle

import fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.chat import clear_cache
from app.db import client, COLLECTION_NAME, VECTOR_SIZE


# ─── CONFIG ────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE", 500))
OVERLAP       = int(os.environ.get("OVERLAP", 50))
BM25_PATH     = os.path.join(os.path.dirname(__file__), "..", "bm25_index.pkl")

print(f"[ingest] Config: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")
print("[ingest] Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ─── PDF READING ───────────────────────────────────────────────────────────────
def load_pdfs(folder: str) -> list[dict]:
    docs = []
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[ingest] ⚠  No PDFs in {folder}")
        return docs
    for filename in pdf_files:
        path = os.path.join(folder, filename)
        print(f"[ingest] Reading: {filename}")
        doc  = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        docs.append({"filename": filename, "text": text})
        print(f"[ingest]   → {len(text)} chars")
    return docs


# ─── CHUNKING ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    PHASE 3 EXPERIMENT: run ingest with different CHUNK_SIZE values:
      CHUNK_SIZE=300 python -c "from app.ingest import ingest_documents; ingest_documents()"
      CHUNK_SIZE=500 python -c "from app.ingest import ingest_documents; ingest_documents()"
      CHUNK_SIZE=800 python -c "from app.ingest import ingest_documents; ingest_documents()"
    Then run evaluate.py each time and record the hit rate in your README table.
    """
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


# ─── MAIN INGEST ───────────────────────────────────────────────────────────────
def ingest_documents():
    if client is None:
        raise RuntimeError(
            "Qdrant is unavailable. Stop the running local Qdrant instance or configure QDRANT_URL."
        )

    clear_cache()

    # Recreate Qdrant collection
    client.recreate_collection(
        collection_name = COLLECTION_NAME,
        vectors_config  = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[ingest] Created Qdrant collection (vector_size={VECTOR_SIZE})")

    docs = load_pdfs(DOCUMENTS_DIR)
    if not docs:
        return

    all_chunks, all_payloads = [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        print(f"[ingest] {doc['filename']} → {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_payloads.append({
                "text":        chunk,
                "source":      doc["filename"],
                "chunk_index": i,
            })

    print(f"[ingest] Embedding {len(all_chunks)} chunks...")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)

    # ── Write to Qdrant ────────────────────────────────────────────────────────
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload=all_payloads[i])
        for i in range(len(all_chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[ingest] ✅  {len(points)} chunks stored in Qdrant")

    # ── Build and save BM25 index ──────────────────────────────────────────────
    # LEARNING: BM25 needs tokenised text — we split on whitespace here.
    # More sophisticated tokenisation (lowercasing, stemming) would improve
    # results but adds complexity. Good enough for Phase 3.
    tokenised = [chunk.lower().split() for chunk in all_chunks]
    bm25      = BM25Okapi(tokenised)

    # Save BM25 + the raw chunks together so retrieval.py can load both
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks, "payloads": all_payloads}, f)
    print(f"[ingest] ✅  BM25 index saved to {BM25_PATH}")
    print("[ingest]    Re-run main.py and try /ask")


if __name__ == "__main__":
    ingest_documents()
