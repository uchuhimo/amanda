from amanda.tools.debugging import insert_debug_op, validator


def test_validator():
    validator.main("vgg16")


def test_insert_debug_op():
    insert_debug_op.main()
