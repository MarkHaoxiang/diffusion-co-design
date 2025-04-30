docker run \
    --volume .:/app \
    --volume /app/.venv \
    -d \
    --name "diffusion-co-design" \
    --gpus=all \
    $(docker build -q .) \
    bash -c "sleep infinity"