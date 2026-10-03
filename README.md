# Resume RAG

## Live app

[Open the RAG Chatbot](https://my-rag-chatbot-anonymous.streamlit.app)

## Retrieval chunk-size experiment

Measured on the six eligible resume PDFs in `documents/` using 30 labeled questions (five per PDF). A question counts as a hit when at least one of the top five retrieved chunks comes from the expected PDF and contains all of that question's expected evidence phrases.

| Chunk size | Hit rate (hit@5) | Notes |
|------------|------------------|-------|
| 300 chars  | 70.0% (21/30)    | Lowest measured retrieval hit rate |
| 500 chars  | 80.0% (24/30)    | Improved over 300 on this corpus |
| 800 chars  | 83.3% (25/30)    | Highest measured retrieval hit rate |

These are retrieval evidence-match scores, not LLM answer-quality scores. The 30 questions are grounded in this local resume set, so results may differ on other documents or evaluation questions. Each run also checks refusal accuracy against 10 questions in `eval/refusal_questions.json`. Refusal checks run through retrieval and the Groq model, so set `GROQ_API_KEY` in `.env` before running the evaluator. Each run builds its index under `qdrant_db/experiments/chunk_<size>` and does not overwrite the regular local index or use the configured remote Qdrant collection.

## Refusal evaluation

| Metric | Result |
|--------|--------|
| Refusal accuracy (chunk size 800) | 100.0% (10/10) |

Re-run any setting with:

```powershell
.\venv\Scripts\python.exe eval/evaluate.py --chunk-size 300
.\venv\Scripts\python.exe eval/evaluate.py --chunk-size 500
.\venv\Scripts\python.exe eval/evaluate.py --chunk-size 800
```
