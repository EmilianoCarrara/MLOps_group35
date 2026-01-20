FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS base

WORKDIR /app

# Disable Weights & Biases in Docker only
ENV WANDB_MODE=disabled

COPY pyproject.toml uv.lock ./

# Remove README.md and LICENSE from myproject.toml
RUN sed -i '/^readme *= *"README.md"/d' pyproject.toml && \
    sed -i '/^license *= *{ *file *= *"LICENSE" *}/d' pyproject.toml

RUN mkdir -p configs
COPY configs/cluster.yaml ./configs/cluster.yaml

RUN mkdir -p data/processed
COPY data/processed/combined.csv ./data/processed/combined.csv

RUN uv sync --frozen --no-install-project

COPY src ./src

RUN uv sync

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.mlops_group35.api:app", "--host", "0.0.0.0", "--port", "8000"]
