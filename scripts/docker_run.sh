docker run \
    --rm \
    --volume .:/app \
    --volume /app/.venv \
    -it \
    --gpus=all \
    $(docker build -q .) \
    "$@"