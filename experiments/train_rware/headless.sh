#!/bin/bash

xvfb-run -a -s "-screen 1 1400x900x24" uv run python main.py "$@"
