# Diffusion Co-Design

When designing multi-agent systems, the environment --- the world agents interact with --- directly impacts behaviour and performance and must not be overlooked. In this work, we consider the \textbf{agent environment co-design} paradigm, an emerging line of research aiming to learn compatible agent policies and environments. An effective co-design framework has utility in various real-world applications such as logistics and energy systems or creative interpretations of an environment like robot configurations and chip design. While existing work has demonstrated success in specialised applications by leveraging reinforcement learning, various challenges restrict widespread adoption --- in particular, complex and combinatorial environment design spaces lead to an exponential number of valid environments to explore. To tackle this challenge, we propose **diffusion co-design** (DiCoDe), a **scalable** co-design methodology leveraging guided diffusion models to design effective environments for collaborative multi-agent tasks. As part of DiCoDe, we introduce projected universal guidance and a training framework that can share knowledge between a reinforcement learning critic function and an environment generator, enabling us to exploit the representational capabilities of diffusion models to generate constraint-satisfying, controllably diverse and highly rewarding environments. We validate DiCoDe on a diverse set of co-design scenarios modelling warehouse management and wind farm control, resulting in policy-environment pairs that exceed state-of-the-art performance in metrics such as time-to-delivery and power output.

![DiCoDe Architecture](./figure.jpg) 

## Setup

Git clone this repository and initialise submodules.

`git submodule update --init --recursive`

We use **uv** to manage dependencies. The easiest way to get set up is by running `./scripts/docker_run.sh`, and entering with an interactive shell. Directly setting up on Linux is possible with `uv sync`, but may require additional system installation of packages (see Dockerfile  for reference).

Due to conflicts with `torch-geometric`, `torch-scatter`, and rendering libraries, additionally commands are required. See `scripts/sync.sh`, or slect between vmas and rware environments with

`./scripts/sync.sh --rware` or `./scripts/sync.sh --vmas`.

## Repository Guided Tour

To train a DiCoDe experiment,

1. Train the exploration diffusion model. The entry point can be found in `experiments/train_env_diffusion`, with dataset generation `setup_scenario.py` and diffusion training `train_diffusion.py`.

2. RL loop. The entry point in `experiments/train_env/main.py`, which constructs a TorchRL training loop using the `Trainer` class found in `diffusion_co_design/common/rl/mappo/trainer.py`. See `diffusion_co_design.common.design.DicodeDesigner` and `diffusion_co_design.common.design.ValueDesigner` for the core logic of our method.

Cleanup and detailed instructions for replicating experiments is a work in progress.