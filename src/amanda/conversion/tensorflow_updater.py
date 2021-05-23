import weakref
from functools import wraps
from typing import Any, MutableMapping, Set

from amanda.import_hook import FunctionUpdater, is_enabled, register_updater

forward_ops_in_graph: MutableMapping[Any, Set[str]] = weakref.WeakKeyDictionary()


def gradients_wrapper(func):
    import tensorflow as tf

    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_enabled():
            tf_graph = tf.get_default_graph()
            forward_ops_in_graph[tf_graph] = set(
                op.name for op in tf_graph.get_operations()
            )
        return func(*args, **kwargs)

    return wrapper


def register_import_hook() -> None:
    register_updater(
        FunctionUpdater(
            module="tensorflow.python.ops.gradients_impl",
            func="gradients",
            decorator=gradients_wrapper,
        )
    )
    register_updater(
        FunctionUpdater(
            module="tensorflow.python.ops.gradients_impl",
            func="gradients_v2",
            decorator=gradients_wrapper,
        )
    )
