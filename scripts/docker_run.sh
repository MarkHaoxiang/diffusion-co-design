docker run \
    --volume .:/app \
    --volume /app/.venv \
    -d \
    --gpus=all \
    $(docker build -q .) \
    bash