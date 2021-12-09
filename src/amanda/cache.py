from contextlib import contextmanager

_cache_enabled = True


@contextmanager
def cache_disabled():
    global _cache_enabled
    prev_cache_enabled = _cache_enabled
    _cache_enabled = False
    try:
        yield
    finally:
        _cache_enabled = prev_cache_enabled


@contextmanager
def cache_enabled():
    global _cache_enabled
    prev_cache_enabled = _cache_enabled
    _cache_enabled = True
    try:
        yield
    finally:
        _cache_enabled = prev_cache_enabled


def is_cache_enabled() -> bool:
    return _cache_enabled
