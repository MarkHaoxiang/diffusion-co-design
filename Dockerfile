FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# MPI
RUN apt-get update && apt-get install -y libopenmpi-dev 

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install project dependencies
# From https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen