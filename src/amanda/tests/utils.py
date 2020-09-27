from amanda.namespace import Namespace, default_namespace

_test_namespace = default_namespace() / Namespace("test")


def test_namespace() -> Namespace:
    return _test_namespace
