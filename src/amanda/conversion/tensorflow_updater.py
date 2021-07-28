import threading
from functools import wraps
from typing import List, Set

from amanda.import_hook import (
    FunctionUpdater,
    InstScopeHook,
    MethodUpdater,
    check_enabled,
    disabled,
    register_inst_scope_hook,
    register_updater,
)
from amanda.tool import get_tools

record_session_run = threading.local()
record_session_run.flag = False


def gradients_wrapper(func):
    import tensorflow as tf

    @wraps(func)
    def wrapper(*args, **kwargs):
        tf_graph = tf.get_default_graph()
        tf_graph._forward_ops = set(op.name for op in tf_graph.get_operations())
        return func(*args, **kwargs)

    return check_enabled(func, wrapper)


def update_graph_in_session(session, tf_graph):
    from tensorflow.python import pywrap_tensorflow as tf_session

    opts = tf_session.TF_NewSessionOptions(
        target=session._target, config=session._config
    )
    try:
        new_raw_session = tf_session.TF_NewSessionRef(tf_graph._c_graph, opts)
        session._session = new_raw_session
    finally:
        tf_session.TF_DeleteSessionOptions(opts)


def session_run_wrapper(func):
    @wraps(func)
    def wrapper(self, fetches, feed_dict=None, options=None, run_metadata=None):
        from amanda.conversion.tensorflow import insert_hooks

        tf_graph = self.graph
        if record_session_run.flag:
            if hasattr(tf_graph, "_session_runs"):
                tf_graph._session_runs.append(
                    [fetches, feed_dict, options, run_metadata]
                )
            else:
                tf_graph._session_runs = [[fetches, feed_dict, options, run_metadata]]
            return func(self, fetches, feed_dict, options, run_metadata)
        else:
            if hasattr(tf_graph, "_forward_ops"):
                forward_ops = tf_graph._forward_ops
            else:
                forward_ops = {op.name for op in tf_graph.get_operations()}
                tf_graph._forward_ops = forward_ops
            if hasattr(tf_graph, "_spec"):
                spec = tf_graph._spec
            else:
                spec = None
            is_hooked = len(tf_graph.get_collection("amanda.is_hooked")) != 0
            if not is_hooked:
                with self.as_default():
                    insert_hooks(tf_graph, spec, get_tools(), forward_ops)
                update_graph_in_session(self, tf_graph)
            if hasattr(tf_graph, "_session_runs"):
                with disabled():
                    for args in tf_graph._session_runs:
                        self.run(*args)
                tf_graph._session_runs.clear()
            return func(self, fetches, feed_dict, options, run_metadata)

    return check_enabled(func, wrapper)


def create_amanda_hook():
    from amanda.conversion.tensorflow import FuncSessionHook

    def after_create_session(session, coord):
        record_session_run.flag = False

    def begin():
        record_session_run.flag = True

    return FuncSessionHook(
        begin=begin,
        after_create_session=after_create_session,
    )


class FilterHook(InstScopeHook):
    def __init__(self) -> None:
        self.begin_ops_list: List[Set[str]] = []
        self.end_ops_list: List[Set[str]] = []
        self.is_enabled_list: List[bool] = []

    @property
    def disabled_ops(self) -> Set[str]:
        disabled_ops: Set[str] = set()
        for begin_ops, end_ops, scope_is_enabled in zip(
            self.begin_ops_list, self.end_ops_list, self.is_enabled_list
        ):
            if scope_is_enabled:
                disabled_ops = disabled_ops - (end_ops - begin_ops)
            else:
                disabled_ops = disabled_ops | (end_ops - begin_ops)
        return disabled_ops

    def begin(self, is_enabled: bool) -> None:
        import tensorflow as tf

        tf_graph = tf.get_default_graph()
        self.begin_ops_list.append(set(op.name for op in tf_graph.get_operations()))
        self.is_enabled_list.append(is_enabled)

    def end(self, is_enabled: bool) -> None:
        import tensorflow as tf

        tf_graph = tf.get_default_graph()
        self.end_ops_list.insert(0, set(op.name for op in tf_graph.get_operations()))


def model_fn_wrapper(model_fn):
    def new_model_fn(features, labels, mode, params, config):
        import tensorflow as tf
        from tensorflow.python.util import function_utils

        model_fn_args = function_utils.fn_args(model_fn)
        kwargs = {}
        if "labels" in model_fn_args:
            kwargs["labels"] = labels
        else:
            if labels is not None:
                raise ValueError(
                    "model_fn does not take labels, but input_fn returns labels."
                )
        if "mode" in model_fn_args:
            kwargs["mode"] = mode
        if "params" in model_fn_args:
            kwargs["params"] = params
        if "config" in model_fn_args:
            kwargs["config"] = config
        tf_graph = tf.get_default_graph()
        input_ops = {op.name for op in tf_graph.get_operations()}
        filter_hook = FilterHook()
        handler = register_inst_scope_hook(filter_hook)
        spec = model_fn(features, **kwargs)
        handler.unregister()
        if hasattr(tf_graph, "_forward_ops"):
            forward_ops = tf_graph._forward_ops
        else:
            forward_ops = set(op.name for op in tf_graph.get_operations())
        tf_graph._forward_ops = forward_ops - input_ops - filter_hook.disabled_ops
        tf_graph._spec = spec
        return spec

    return new_model_fn


def train_wrapper(func):
    @wraps(func)
    def wrapper(
        self,
        input_fn,
        hooks=None,
        steps=None,
        max_steps=None,
        saving_listeners=None,
    ):
        original_model_fn = self._model_fn
        self._model_fn = model_fn_wrapper(self._model_fn)
        result = func(
            self,
            input_fn,
            [create_amanda_hook()] if hooks is None else [*hooks, create_amanda_hook()],
            steps,
            max_steps,
            saving_listeners,
        )
        self._model_fn = original_model_fn
        return result

    return check_enabled(func, wrapper)


def predict_wrapper(func):
    @wraps(func)
    def wrapper(
        self,
        input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True,
    ):
        original_model_fn = self._model_fn
        self._model_fn = model_fn_wrapper(self._model_fn)
        result = func(
            self,
            input_fn,
            predict_keys,
            [create_amanda_hook()] if hooks is None else [*hooks, create_amanda_hook()],
            checkpoint_path,
            yield_single_examples,
        )
        self._model_fn = original_model_fn
        return result

    return check_enabled(func, wrapper)


def evaluate_wrapper(func):
    @wraps(func)
    def wrapper(
        self,
        input_fn,
        steps=None,
        hooks=None,
        checkpoint_path=None,
        name=None,
    ):
        original_model_fn = self._model_fn
        self._model_fn = model_fn_wrapper(self._model_fn)
        result = func(
            self,
            input_fn,
            steps,
            [create_amanda_hook()] if hooks is None else [*hooks, create_amanda_hook()],
            checkpoint_path,
            name,
        )
        self._model_fn = original_model_fn
        return result

    return check_enabled(func, wrapper)


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
    register_updater(
        MethodUpdater(
            module="tensorflow_estimator.python.estimator.estimator",
            cls="Estimator",
            method="train",
            decorator=train_wrapper,
        )
    )
    register_updater(
        MethodUpdater(
            module="tensorflow_estimator.python.estimator.estimator",
            cls="Estimator",
            method="predict",
            decorator=predict_wrapper,
        )
    )
    register_updater(
        MethodUpdater(
            module="tensorflow_estimator.python.estimator.estimator",
            cls="Estimator",
            method="evaluate",
            decorator=evaluate_wrapper,
        )
    )
