# Dockerfile — Phase 7: containerise the RAG app
#
# WHAT YOU LEARN HERE:
# - Multi-stage thinking: why we use python:3.11-slim not python:3.11
# - Layer caching: why requirements.txt is copied BEFORE the app code
# - Non-root user: basic container security practice
# - CMD vs ENTRYPOINT: CMD is the default, overridable at runtime
#
# BUILD:  docker build -t my-rag .
# RUN:    docker run -p 8000:8000 --env-file .env my-rag

# ── Base image ─────────────────────────────────────────────────────────────────
# python:3.11-slim = Python 3.11 on Debian with minimal extras (~150 MB).
# Avoids the full python:3.11 image (~900 MB) which includes compilers,
# build tools, and dozens of libraries you don't need at runtime.
FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
# PyMuPDF (fitz) needs libGL and libglib at runtime.
# --no-install-recommends keeps the layer small.
# rm -rf /var/lib/apt/lists/* deletes the apt cache — another size reduction.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps BEFORE copying app code ────────────────────────────────
# LEARNING: Docker builds in layers. Each instruction is a cached layer.
# If you copy ALL files first and then pip install, any code change
# invalidates the pip layer and forces a full reinstall (~3 min).
# Copying requirements.txt first means pip layer is only rebuilt when
# requirements.txt changes — not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy app code ──────────────────────────────────────────────────────────────
# .dockerignore (see below) prevents copying venv/, qdrant_db/, bm25_index.pkl
# and other local artefacts that don't belong in the image.
COPY . .

# ── Non-root user ──────────────────────────────────────────────────────────────
# LEARNING: Running as root inside a container is a security risk.
# If a vulnerability lets someone escape the container, they'd be root
# on the host. This is a simple, standard mitigation.
RUN mkdir -p /app/qdrant_db && useradd -m appuser && chown -R appuser /app
USER appuser

# ── Port ───────────────────────────────────────────────────────────────────────
# EXPOSE documents which port the container listens on.
# It does NOT publish the port — that's done with -p at docker run time.
EXPOSE 8000

# ── Startup ────────────────────────────────────────────────────────────────────
# --host 0.0.0.0 : listen on all interfaces inside the container
#                  (127.0.0.1 would only accept connections from inside the container)
# --workers 1    : single worker is fine for a portfolio project
# no --reload    : reload is for development only, not production
CMD ["python", "main.py"]
