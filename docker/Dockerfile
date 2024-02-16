FROM python:3.10.0-slim-buster

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh &&
RUN bash -c "uv venv && source .venv/bin/activate && uv pip install -r requirements.txt"

CMD mkdir -p /workspace
WORKDIR /workspace
