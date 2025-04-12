import os
from os.path import join

# Simpler to use this than environment variables for now
_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR = join(_FILE_DIRECTORY, "../bin/conf")
OUTPUT_DIR = join(os.environ["HOME"], ".diffusion_co_design")
EXPERIMENT_DIR = join(OUTPUT_DIR, "experiments")
