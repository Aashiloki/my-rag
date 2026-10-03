"""
main.py — Phase 3: FastAPI with /ingest, /ask, /evaluate, /stats

NEW IN PHASE 3:
- /ask now returns citations, token counts, cache_hit flag
- /evaluate endpoint runs the eval set (so you can trigger it via HTTP)
- /stats endpoint returns token usage and cache hit counts
- .env file support via python-dotenv

HOW TO RUN:
  # Windows PowerShell
  $env:GROQ_API_KEY="gsk_your_key"
  uvicorn main:app --reload

  # Or create a .env file (easier):
  echo GROQ_API_KEY=gsk_your_key > .env
  uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import sys, os

load_dotenv()   # reads .env file if it exists — no more $env: every session

sys.path.insert(0, os.path.dirname(__file__))

from app.ingest    import ingest_documents
from app.retrieval import retrieve
from app.chat      import ask, token_stats

app = FastAPI(
    title       = "My RAG Chatbot — Phase 3",
    description = "Hybrid retrieval + re-ranking + citations + evaluation",
    version     = "0.3.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODELS ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k:    int = 5

class AskResponse(BaseModel):
    question:          str
    answer:            str
    sources:           list[str]
    is_refusal:        bool
    cache_hit:         bool
    prompt_tokens:     int
    completion_tokens: int

class IngestResponse(BaseModel):
    message: str

class StatsResponse(BaseModel):
    total_queries:           int
    total_prompt_tokens:     int
    total_completion_tokens: int
    cache_hits:              int
    estimated_cost_usd:      float   # rough estimate at Groq free tier

class EvalResponse(BaseModel):
    hit_rate:     float
    refusal_rate: float
    message:      str


# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":    "ok",
        "version":   "phase3",
        "endpoints": ["/ingest", "/ask", "/evaluate", "/stats", "/docs"]
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load PDFs → chunk → embed → store in Qdrant + build BM25 index."""
    try:
        ingest_documents()
        return {"message": "Ingestion complete. BM25 + Qdrant indexes built."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Hybrid retrieval + cross-encoder re-ranking + grounded LLM answer.

    Returns the answer with inline [Source N] citations, plus:
    - is_refusal: True if context was insufficient
    - cache_hit: True if this exact question was answered before
    - token counts for cost tracking
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(request.question, top_k=request.top_k)

    try:
        result = ask(request.question, chunks)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    sources = list({c["source"] for c in chunks})

    return AskResponse(
        question          = request.question,
        answer            = result["answer"],
        sources           = sources,
        is_refusal        = result["is_refusal"],
        cache_hit         = result["cache_hit"],
        prompt_tokens     = result["prompt_tokens"],
        completion_tokens = result["completion_tokens"],
    )


@app.post("/evaluate", response_model=EvalResponse)
def evaluate():
    """
    Runs the 30-question evaluation set.
    Reports retrieval hit rate and refusal accuracy.

    LEARNING: Having an /evaluate endpoint means you can trigger an
    eval run from CI/CD (GitHub Actions) automatically when you push
    changes — you'd never ship a retrieval regression to production.
    That's the professional version of what you're doing manually now.
    """
    try:
        from eval.evaluate import run_evaluation
        hit_rate, refusal_rate = run_evaluation()
        return EvalResponse(
            hit_rate     = round(hit_rate, 1),
            refusal_rate = round(refusal_rate, 1),
            message      = f"Hit rate: {hit_rate:.1f}% | Refusal accuracy: {refusal_rate:.1f}%",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
def stats():
    """
    Returns token usage since server started.
    Rough cost estimate based on Groq's free tier (llama-3.1-8b pricing).
    """
    total_tokens = (
        token_stats["total_prompt_tokens"] +
        token_stats["total_completion_tokens"]
    )
    # llama-3.1-8b on Groq: ~$0.05 per million tokens (approximate)
    cost = total_tokens * 0.05 / 1_000_000

    return StatsResponse(
        total_queries           = token_stats["total_queries"],
        total_prompt_tokens     = token_stats["total_prompt_tokens"],
        total_completion_tokens = token_stats["total_completion_tokens"],
        cache_hits              = token_stats["cache_hits"],
        estimated_cost_usd      = round(cost, 6),
    )


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting Phase 3 RAG server...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
