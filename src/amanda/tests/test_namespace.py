import pytest

from amanda import Namespace
from amanda.namespace import get_base_name, get_namespace, is_qualified


@pytest.fixture
def namespace():
    return Namespace("level1/level2")


def test_qualified(namespace):
    assert namespace.qualified("level3") == "/level1/level2/level3"


def test_belong_to(namespace):
    assert namespace.belong_to(Namespace("level1"))


def test_truediv(namespace):
    assert (namespace / "level3").namespace == "level1/level2/level3"


def test_eq(namespace):
    assert namespace == Namespace("level1/level2")


def test_repr(namespace):
    assert repr(namespace) == "Namespace(level1/level2)"


def test_is_qualified():
    assert is_qualified("/level1/level2")


def test_get_namespace():
    assert get_namespace("/level1/level2") == "level1"
    assert get_namespace("level2") == ""


def test_get_base_name():
    assert get_base_name("/level1/level2") == "level2"
    assert get_base_name("level2") == "level2"
