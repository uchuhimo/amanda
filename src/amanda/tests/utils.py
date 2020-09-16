import os
from pathlib import Path

from amanda.namespace import Namespace, default_namespace

_test_namespace = default_namespace() / Namespace("test")


def test_namespace() -> Namespace:
    return _test_namespace


def root_dir() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir).parents[2]
