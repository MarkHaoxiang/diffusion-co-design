FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
RUN apt-get update

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project
COPY . /app
WORKDIR /app

# UV dependencies
RUN uv sync --frozen

RUN git clone 