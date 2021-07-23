import threading
import weakref
from functools import wraps
from typing import Any, List, MutableMapping, Set

from amanda.import_hook import (
    FunctionUpdater,
    MethodUpdater,
    is_enabled,
    register_updater,
)

forward_ops_in_graph: MutableMapping[Any, Set[str]] = weakref.WeakKeyDictionary()
session_run_in_graph: MutableMapping[Any, List[List[Any]]] = weakref.WeakKeyDictionary()
record_session_run = threading.local()
record_session_run.flag = False


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


def session_run_wrapper(func):
    import tensorflow as tf

    @wraps(func)
    def wrapper(self, fetches, feed_dict=None, options=None, run_metadata=None):
        if is_enabled() and record_session_run.flag:
            tf_graph = tf.get_default_graph()
            if tf_graph in session_run_in_graph:
                session_run_in_graph[tf_graph].append(
                    [fetches, feed_dict, options, run_metadata]
                )
            else:
                session_run_in_graph[tf_graph] = [
                    [fetches, feed_dict, options, run_metadata]
                ]
        return func(self, fetches, feed_dict, options, run_metadata)

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
    register_updater(
        MethodUpdater(
            module="tensorflow.python.client.session",
            cls="Session",
            method="run",
            decorator=session_run_wrapper,
        )
    )
