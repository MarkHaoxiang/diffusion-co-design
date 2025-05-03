FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y libopenmpi-dev xvfb git libfontconfig1

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV DIFFUSION_CO_DESIGN_WDIR=/app/.diffusion_co_design

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