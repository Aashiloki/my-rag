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

GitHub Actions runs unit tests, rebuilds a fresh local index from `documents/`, enforces the retrieval and refusal thresholds, then builds and scans the Docker image. Add a repository Actions secret named `GROQ_API_KEY` for refusal evaluation. Forked pull requests do not receive repository secrets and therefore cannot run the live Groq evaluation.

## CI/CD pipeline stages and blocking gates

The workflow is intentionally strict and fails fast:

1. `test` runs lint- and unit-level checks. If it fails, the pipeline stops before evaluation.
2. `eval` rebuilds the corpus index and runs `python eval/evaluate.py --min-hit-rate 70 --min-refusal-accuracy 100`. It blocks on retrieval regressions or refusal behavior regressions.
3. `build` creates the Docker image and runs a Trivy scan. Critical vulnerabilities block deployment.
4. `deploy` only runs on `main` after the previous stages succeed, then verifies the Railway health endpoint at `/health`.

This means the deployment job is gated by both product-quality checks and security scanning. The deploy step also validates environment configuration before it touches Railway.

## Honest baseline and thresholds

The evaluation is intentionally conservative. The current local baseline after the fixes is approximately:

- retrieval hit rate: 85%
- refusal accuracy: 100%

The production gate in CI is lower than the current healthy baseline, but still strict enough to catch obvious regressions:

- minimum hit rate: 70%
- minimum refusal accuracy: 100%

The repository does not lower the threshold to make a score look green. A change is only considered acceptable if it meets the gate without masking a real retrieval problem.

## Trivy findings and fixes

The Docker build scan was flagging critical vulnerabilities in the Python ML stack, specifically the `torch` and `sentence-transformers` dependency chain. The fix was to upgrade the stack to a compatible tested combination that still satisfied the project requirements and building in CI/Docker:

- upgraded `sentence-transformers` to a newer supported version
- pinned a compatible CPU wheel for `torch` (`2.7.1+cpu` in the Docker/CI path)
- kept the dependency set consistent across local, CI, and container builds

This was not a case of ignoring the vulnerability report. The project moved to a supported compatible stack and kept the security gate enabled.

## Rollback runbook

If a deployment causes a production problem:

1. Stop the rollout in GitHub by disabling the workflow or reverting the last merge.
2. Revert the offending commit in `main` with `git revert <sha>` or by rolling back to the last known-good release.
3. Redeploy the previous image or previous application version from Railway.
4. Re-run the evaluation locally or in CI to confirm that the regression is gone before re-enabling deployment.
5. After the fix is validated, open a PR with the root cause and the mitigation documented, then merge once the checks pass again.

For operational safety, keep the environment protection rules in place and only deploy from the same guarded workflow path after the quality gates pass again.

On pushes to `main`, the workflow deploys to Railway after all checks pass and verifies `/health`. Configure `RAILWAY_TOKEN` as a secret in the `production` environment, and set the Actions variables `RAILWAY_SERVICE` and `APP_URL`. Disable Railway's automatic GitHub deployment trigger so it does not race this workflow.