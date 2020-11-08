import contextlib
from typing import Any, List

from amanda.lang import replace_all_refs

stack: List[Any] = []


@contextlib.contextmanager
def use_replace_all_refs(default):
    stack.clear()
    stack.append(default)
    try:
        yield default
    finally:
        assert stack[-1] is default


def use_replace_all_refs2(default, fn):
    stack.clear()
    stack.append(default)
    try:
        fn(default)
    finally:
        assert stack[-1] is default


def test_replace_all_refs():
    with use_replace_all_refs([1]) as x:
        y = [2]
        replace_all_refs(x, y)


def test_replace_all_refs2():
    def fn(x):
        y = [2]
        replace_all_refs(x, y)

    use_replace_all_refs2([1], fn)
