import pytest

from amanda.tools.debugging import insert_debug_op, validator


def test_validator():
    validator.main("vgg16")


@pytest.mark.skip
def test_insert_debug_op():
    insert_debug_op.main()
