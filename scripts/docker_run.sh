docker run \
    --volume .:/app \
    --volume /app/.venv \
    --volume "${DIFFUSION_CO_DESIGN_WDIR:-$HOME/.diffusion_co_design}:/app/.diffusion_co_design" \
    -d \
    --name "diffusion-co-design" \
    --gpus=all \
    $(docker build -q .) \
    bash -c "sleep infinity"