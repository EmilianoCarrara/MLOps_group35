FROM ghcr.io/astral-sh/uv:python3.11-alpine AS base

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv sync --frozen --no-install-project

COPY src ./src

RUN uv sync

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "mlops_group35.api:app", "--host", "0.0.0.0", "--port", "8000"]
