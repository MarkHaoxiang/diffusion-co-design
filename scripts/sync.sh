#!/bin/bash

ENV_TYPE=""
PYPROJECT="$(dirname "$0")/../pyproject.toml"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --rware)
            ENV_TYPE="rware"
            ;;
        --vmas)
            ENV_TYPE="vmas"
            ;;
        *)
            echo "Usage: $0 [--rware | --vmas]"
            exit 1
            ;;
    esac
    shift
done

# Validate input
if [[ -z "$ENV_TYPE" ]]; then
    echo "Usage: $0 [--rware | --vmas]"
    exit 1
fi

if [[ "$ENV_TYPE" == "rware" ]]; then
    sed -i 's|^default-groups = \["dev", "vmas"\]|# default-groups = ["dev", "vmas"]|' "$PYPROJECT"
    sed -i 's|^# default-groups = \["dev", "rware"\]|default-groups = ["dev", "rware"]|' "$PYPROJECT"
    sed -i 's|^# *\(rware = { workspace = true }\)|\1|' "$PYPROJECT"

elif [[ "$ENV_TYPE" == "vmas" ]]; then
    sed -i 's|^default-groups = \["dev", "rware"\]|# default-groups = ["dev", "rware"]|' "$PYPROJECT"
    sed -i 's|^# default-groups = \["dev", "vmas"\]|default-groups = ["dev", "vmas"]|' "$PYPROJECT"
    sed -i 's|^\(rware = { workspace = true }\)|# \1|' "$PYPROJECT"
fi


uv sync
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html --no-build-isolation
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html --no-build-isolation