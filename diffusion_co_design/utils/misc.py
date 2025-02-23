import os
import re


def get_latest_model(dir: str, prefix: str) -> str:
    files = os.listdir(dir)
    target_files = [f for f in files if re.match(rf"{prefix}\d+\.pt", f)]
    latest = max(target_files, key=lambda x: int(re.search(r"\d+", x).group()))
    return os.path.join(dir, latest)
