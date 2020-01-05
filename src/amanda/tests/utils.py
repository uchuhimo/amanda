import os
from pathlib import Path


def root_dir() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir).parents[2]
