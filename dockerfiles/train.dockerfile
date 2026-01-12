FROM ghcr.io/astral-sh/uv:python3.11-alpine AS base

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv sync --frozen --no-install-project

COPY src ./src

RUN uv sync

ENTRYPOINT ["uv", "run", "python", "src/mlops_group35/train.py"]