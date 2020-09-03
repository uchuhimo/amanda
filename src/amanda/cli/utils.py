import importlib
from typing import Any, Callable


def import_from_name(name: str) -> Callable[..., Any]:
    module_name, basic_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, basic_name)
