FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# ── Install uv (fast Python package manager) ─────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

WORKDIR /app

# ── Install dependencies (cached layer — only re-runs when lock changes) ─────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Copy application code ────────────────────────────────────────────────────
COPY . .

# ── Hugging Face Spaces expects port 7860 ────────────────────────────────────
EXPOSE 7860

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
