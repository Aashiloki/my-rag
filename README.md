# Medicaps RAG

## Live app

[Open the RAG chatbot](https://my-rag-chatbot-anonymous.streamlit.app)

[Railway API](https://my-rag-production.up.railway.app)

## Documents

The corpus in `documents/` contains Medicaps University academic, examination, conduct, syllabus, and prospectus PDFs. Some files are scans without extractable text, so they contribute no searchable content until OCR is added.

## Evaluation

The retrieval evaluation uses 20 answerable questions from `eval/questions.json`. The separate `eval/refusal_questions.json` contains 10 questions that should be refused because their answers are not in the corpus. Retrieval hit rate checks whether one of the top-five retrieved chunks contains every expected keyword. Refusal accuracy checks whether the assistant returns its refusal response.

Build the local index before evaluating. Refusal checks call Groq, so set `GROQ_API_KEY` in your environment first.

```powershell
python -m unittest discover -s tests -v
python -m app.ingest
python eval/evaluate.py --min-hit-rate 70 --min-refusal-accuracy 100
docker build -t medicaps-rag .
```

The initial CI gates are a 70% retrieval hit rate and 100% refusal accuracy. The evaluator prints both measured rates and exits with a nonzero status if a requested minimum is missed. Run ingestion again after changing `CHUNK_SIZE`; the evaluator measures the index currently on disk.

GitHub Actions installs dependencies, runs the tests, builds a fresh local index from `documents/`, runs the gated evaluation, and builds the Docker image. Add a repository Actions secret named `GROQ_API_KEY` for the refusal evaluation. Forked pull requests do not receive repository secrets and therefore cannot run the live Groq evaluation.

The workflow validates changes but does not deploy them. Railway's automatic deployment is configured outside this repository; disable that trigger if deployments must wait for GitHub Actions checks.