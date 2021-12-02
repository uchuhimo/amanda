from contextlib import contextmanager

from amanda.threading import ThreadLocalStack

_cache_enabled = ThreadLocalStack()


@contextmanager
def cache_disabled():
    _cache_enabled.push(False)
    try:
        yield
    finally:
        _cache_enabled.pop()


@contextmanager
def cache_enabled():
    _cache_enabled.push(True)
    try:
        yield
    finally:
        _cache_enabled.pop()


def is_cache_enabled() -> bool:
    return _cache_enabled.top() or _cache_enabled.top() is None
