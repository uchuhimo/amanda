from amanda.tools.debugging import (
    insert_debug_op_adhoc_pytorch,
    insert_debug_op_adhoc_tensorflow,
    insert_debug_op_pytorch,
    insert_debug_op_tensorflow,
    validator,
)


def test_validator():
    validator.main("vgg16", validator.modify_graph)


def test_validator_with_tf_func():
    validator.main("vgg16", validator.modify_graph_with_tf_func)


def test_insert_debug_op_tensorflow():
    insert_debug_op_tensorflow.main()


def test_insert_debug_op_pytorch():
    insert_debug_op_pytorch.main()


def test_insert_debug_op_adhoc_tensorflow():
    insert_debug_op_adhoc_tensorflow.main()


def test_insert_debug_op_adhoc_pytorch():
    insert_debug_op_adhoc_pytorch.main()
