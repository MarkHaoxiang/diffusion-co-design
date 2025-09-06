# Synchronise DCD setup files between different servers
import os
import argparse
import subprocess
from diffusion_co_design.common import get_latest_model, OUTPUT_DIR


def main(remote, remote_dir):
    to_sync = []

    environments = os.listdir(OUTPUT_DIR)
    environments = [
        e for e in environments if not e.startswith(".") and e != "experiments"
    ]

    for environnment in environments:
        diffusion_dir = os.path.join(OUTPUT_DIR, environnment, "diffusion")
        for root, dirs, _ in os.walk(diffusion_dir):
            if not dirs:
                diffusion_model = get_latest_model(root, "model")
                to_sync.append(diffusion_model)

        # TODO markli: Maybe we should avoid syncing the dataset in scenario
        scenario_dir = os.path.join(OUTPUT_DIR, environnment, "scenario")
        to_sync.append(scenario_dir)

    for path in to_sync:
        rel_path = os.path.relpath(path, OUTPUT_DIR)
        remote_path = os.path.join(remote_dir, rel_path)

        # ensure parent dir exists on remote

        subprocess.run(
            ["ssh", remote, f"mkdir -p {remote_path}"],
            check=True,
        )

        # copy file or dir
        subprocess.run(
            ["scp", "-r", path, f"{remote}:{remote_path}"],
            check=True,
        )

        # Print commands to test
        # print(f"ssh {remote} 'mkdir -p {os.path.dirname(remote_path)}'")
        # print(f"scp -r {path} {remote}:{remote_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize DCD setup files between different servers"
    )
    parser.add_argument(
        "--remote",
        required=True,
        help="Remote host in the form user@host",
    )
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Base directory on the remote server",
    )

    args = parser.parse_args()
    main(args.remote, args.remote_dir)
