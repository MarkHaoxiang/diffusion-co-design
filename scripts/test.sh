#!/bin/bash

ACCEPT_MODE=0

while getopts "e" opt; do
    case "$opt" in
        e) ACCEPT_MODE=1 ;;
        *) echo "Usage: $0 [-e]" >&2; exit 1 ;;
    esac
done

if [[ $ACCEPT_MODE -eq 1 ]]; then
    export EXPECTTEST_ACCEPT=1
fi

uv run pytest tests