import os
from collections import defaultdict
from .util import scp

REMOTE = "excalibur"  # or morgana
source = f"hxl23@{REMOTE}.cl.cam.ac.uk"

EXPERIMENTS: list[tuple[str, str]] = []

if __name__ == "__main__":
    exp_dict = defaultdict(list)
    for experiment, source in EXPERIMENTS:
        exp_dict[experiment].append(source)

    for experiment, sources in exp_dict.items():
        for i, source in enumerate(sources):
            remote_path = source.replace("/app/", "/local/scratch/hxl23/")

            for file in ("policy_3999.pt", "critic_3999.pt", "designer_3999.pt"):
                target = os.path.join("downloaded_checkpoints", experiment)
                os.makedirs(target, exist_ok=True)
                target = os.path.join(target, f"{REMOTE}_{i}")
                os.makedirs(target, exist_ok=True)
                path = os.path.join(remote_path, file)
                scp(source, path, target)
