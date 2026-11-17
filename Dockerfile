# --- Етап 1: Builder (встановлення залежностей та пакета за uv.lock) ---
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev


# --- Етап 2: Runtime (мінімальний образ від non-root користувача) ---
FROM python:3.12-slim AS runtime

RUN groupadd --gid 10001 findex \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash findex

WORKDIR /app

COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/src /app/src
COPY --chown=findex:findex data/corpus /app/data/corpus

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH="/app/data/index.bin" \
    CORPUS_DIR="/app/data/corpus" \
    EMBEDDINGS_PATH="/app/data/embeddings.npy" \
    PORT="8000"

RUN mkdir -p /app/data && chown -R findex:findex /app/data

USER findex

RUN findex index /app/data/corpus --out /app/data/index.bin --executor serial --workers 1 \
    && findex embed /app/data/corpus --out /app/data/embeddings.npy

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", "findex serve --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]