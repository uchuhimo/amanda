import weakref
from dataclasses import dataclass
from functools import wraps
from typing import Any, List, MutableMapping, NamedTuple, OrderedDict, Set

from amanda.import_hook import (
    InstScopeHook,
    check_enabled,
    disabled,
    is_enabled,
    register_inst_scope_hook,
)
from amanda.tool import get_tools


def update_node(tf_graph, node, updates):
    name = node.name
    if ":" in name:
        if name in updates["updated_outputs"]:
            updated_node_name = updates["updated_outputs"][name]
            while updated_node_name in updates["updated_outputs"]:
                updated_node_name = updates["updated_outputs"][updated_node_name]
            return tf_graph.get_tensor_by_name(updated_node_name)
        else:
            return tf_graph.get_tensor_by_name(name)
    else:
        if name in updates["updated_ops"]:
            updated_node_name = updates["updated_ops"][name]
            while updated_node_name in updates["updated_ops"]:
                updated_node_name = updates["updated_ops"][updated_node_name]
            return tf_graph.get_operation_by_name(updated_node_name)
        else:
            return tf_graph.get_operation_by_name(name)


def update_fetches(tf_graph, fetches, updates):
    import tensorflow as tf

    if isinstance(fetches, list):
        for index, element in enumerate(fetches):
            new_element = update_fetches(tf_graph, element, updates)
            fetches[index] = new_element
    elif isinstance(fetches, tuple):
        fetches_list = update_fetches(tf_graph, list(fetches), updates)
        fetches = tuple(fetches_list)
    elif isinstance(fetches, dict):
        for key, value in fetches.items():
            new_value = update_fetches(tf_graph, value, updates)
            fetches[key] = new_value
    elif isinstance(fetches, OrderedDict):
        for key, value in fetches.items():
            new_value = update_fetches(tf_graph, value, updates)
            fetches[key] = new_value
    elif isinstance(fetches, NamedTuple):
        fetches_list = update_fetches(tf_graph, list(fetches), updates)
        fetches = type(fetches)._make(fetches_list)
    elif isinstance(
        fetches, (tf.Operation, tf.Tensor, tf.sparse.SparseTensor, tf.Variable)
    ):
        new_fetches = update_node(tf_graph, fetches, updates)
        fetches = new_fetches
    elif isinstance(fetches, str):
        name = fetches
        if ":" in name:
            node = tf_graph.get_tensor_by_name(name)
        else:
            node = tf_graph.get_operation_by_name(name)
        new_node = update_node(tf_graph, node, updates)
        fetches = new_node.name
    else:
        raise RuntimeError(f"{fetches} is unsupported as a part of fetches")
    return fetches


def update_graph(tf_graph, updates):
    before_op_update = updates["before_op_update"]
    after_op_update = updates["after_op_update"]
    for tf_op_name, index, new_input_name in before_op_update:
        tf_op = tf_graph.get_operation_by_name(tf_op_name)
        new_input = tf_graph.get_tensor_by_name(new_input_name)
        tf_op._update_input(index, new_input)

    for (
        tf_op_name,
        index,
        output_name,
        new_output_name,
        new_op_names,
    ) in after_op_update:
        tf_op = tf_graph.get_operation_by_name(tf_op_name)
        output = tf_graph.get_tensor_by_name(output_name)
        new_output = tf_graph.get_tensor_by_name(new_output_name)
        for consumer in output.consumers():
            if consumer.name in new_op_names:
                continue
            for input_index, input in enumerate(consumer.inputs):
                if input == output:
                    consumer._update_input(input_index, new_output)
                    break


def remove_op_from_graph(graph, op):
    from amanda.conversion.amanda_tf_pybind import remove_op

    del graph._nodes_by_id[op._id]
    del graph._nodes_by_name[op.name]
    del graph._names_in_use[op.name.lower()]
    remove_op(graph._c_graph, op._c_op)


def save_op(filename_tensor, saveables):
    from tensorflow.python.training.saver import BaseSaverBuilder

    saver = BaseSaverBuilder()
    return saver.save_op(filename_tensor, saveables)


def restore_op(filename_tensor, saveables):
    from tensorflow.python.training.saver import BaseSaverBuilder

    saver = BaseSaverBuilder()
    return saver.bulk_restore(filename_tensor, saveables, 0, False)


def get_saveable(iterator):
    from tensorflow.python.data.ops.iterator_ops import _IteratorSaveable

    return _IteratorSaveable(iterator.outputs[0], iterator.name)


def get_iterators(graph):
    return [
        op for op in graph.get_operations() if op.type in ["Iterator", "IteratorV2"]
    ]


def get_iterator_inits(graph):
    return [op for op in graph.get_operations() if op.type == "MakeIterator"]


def get_all_variables():
    import tensorflow as tf

    return tf.global_variables() + tf.local_variables()


@dataclass
class TensorFlowAdapter:
    last_id: int = None
    instrumented_graph: Any = None
    updates: Any = None

    def get_graph(self, session):
        return session._graph

    def set_graph(self, session, graph):
        session._graph = graph

    def clone(self, graph):
        import tensorflow as tf
        from tensorflow.python.framework.meta_graph import import_scoped_meta_graph

        original_last_id = graph._last_id
        meta_graph_def = tf.train.export_meta_graph(graph=graph)
        new_graph = tf.Graph()
        with new_graph.as_default():
            import_scoped_meta_graph(meta_graph_def)
        if original_last_id != new_graph._last_id:
            raise RuntimeError(
                "clone introduces new ops: "
                + str(new_graph.get_operations()[original_last_id:])
            )
        return new_graph

    def extract(self, session):
        import tensorflow as tf

        snapshot = {}
        tf_graph_finalized = session.graph._finalized
        session.graph._finalized = False
        original_last_id = session.graph._last_id
        with session.graph.as_default(), session.as_default():
            all_variables = get_all_variables()
            uninitialized_variables = session.run(
                tf.report_uninitialized_variables(get_all_variables())
            )
            initialized_variables = set(
                map(lambda variable: variable.op.name, all_variables)
            ) - set(map(lambda x: x.decode(), uninitialized_variables))
            variable_values = session.run(
                {variable: variable + ":0" for variable in initialized_variables}
            )
        snapshot["variables"] = variable_values

        # _, file_name = tempfile.mkstemp()
        # filename_tensor = tf.convert_to_tensor(file_name)
        # saveables = [
        #     get_saveable(iterator) for iterator in get_iterators(session.graph)
        # ]
        # op = save_op(filename_tensor, saveables)
        # session.run(op)
        # snapshot["iterators"] = file_name

        inserted_ops = [
            session.graph._nodes_by_id[op_id]
            for op_id in range(original_last_id + 1, session.graph._last_id + 1)
        ]
        for op in inserted_ops:
            remove_op_from_graph(session.graph, op)
        session.graph._next_id_counter = original_last_id
        if original_last_id != session.graph._last_id:
            raise RuntimeError(
                "extract introduces new ops: "
                + str(session.graph.get_operations()[original_last_id:])
            )

        if tf_graph_finalized:
            session.graph._finalized = True
        return session.graph, snapshot

    def load(self, session, graph, snapshot):
        from tensorflow.python import pywrap_tensorflow as tf_session

        opts = tf_session.TF_NewSessionOptions(
            target=session._target, config=session._config
        )
        try:
            new_raw_session = tf_session.TF_NewSessionRef(graph._c_graph, opts)
            tf_session.TF_CloseSession(session._session)
            tf_session.TF_DeleteSession(session._session)
            session._session = new_raw_session
        finally:
            tf_session.TF_DeleteSessionOptions(opts)
        session._graph = graph
        with graph.as_default(), session.as_default():
            src_variable_values = snapshot["variables"]
            dst_variable_names = set(
                map(lambda variable: variable.op.name, get_all_variables())
            ).intersection(set(src_variable_values.keys()))
            dst_variables = [
                variable
                for variable in get_all_variables()
                if variable.op.name in dst_variable_names
            ]
            session.run(
                [variable.initializer for variable in dst_variables],
                feed_dict={
                    variable.initializer.inputs[1]: src_variable_values[
                        variable.op.name
                    ]
                    for variable in dst_variables
                },
            )
            session.run(get_iterator_inits(graph))

    def is_graph_updated(self, graph):
        if self.last_id is not None:
            return graph._last_id > self.last_id
        else:
            return True

    def save_prev_graph(self, graph):
        self.last_id = graph._last_id

    def update_fetches(self, fetches, updates):
        return update_fetches(self.instrumented_graph, fetches, updates)

    def execute_routines(self, session):
        from amanda.conversion.tf import collect_actions

        graph = session.graph
        with graph.as_default(), session.as_default():
            actions = collect_actions(graph, get_tools())
        return actions

    def execute_actions(self, graph, actions):
        from amanda.conversion.tf import apply_actions

        with graph.as_default():
            updates = apply_actions(graph, actions)
        update_graph(graph, updates)
        return graph, updates

    def instrument(self, session, graph):
        if self.instrumented_graph is not None:
            self.set_graph(session, self.instrumented_graph)
        instrumented_graph, snapshot = self.extract(session)
        self.load(session, self.clone(graph), snapshot)
        actions = self.execute_routines(session)
        instrumented_graph, snapshot = self.extract(session)
        instrumented_graph, updates = self.execute_actions(instrumented_graph, actions)
        self.load(session, instrumented_graph, snapshot)
        return instrumented_graph, updates


_adapters: MutableMapping[Any, TensorFlowAdapter] = weakref.WeakKeyDictionary()


def get_adapter(session) -> TensorFlowAdapter:
    if session not in _adapters:
        _adapters[session] = TensorFlowAdapter()
    return _adapters[session]


def session_run_wrapper(func):
    @wraps(func)
    def wrapper(self, fetches, feed_dict=None, options=None, run_metadata=None):
        session = self
        with disabled():
            adapter = get_adapter(session)
            graph = adapter.get_graph(session)
            if adapter.is_graph_updated(graph):
                instrumented_graph, updates = adapter.instrument(session, graph)
                adapter.instrumented_graph = instrumented_graph
                adapter.updates = updates
            instrumented_graph = adapter.instrumented_graph
            updates = adapter.updates
            adapter.set_graph(session, instrumented_graph)
            fetches = adapter.update_fetches(fetches, updates)
            result = session.run(fetches, feed_dict, options, run_metadata)
            adapter.set_graph(session, graph)
            adapter.save_prev_graph(graph)
            return result

    return check_enabled(func, wrapper)


def create_amanda_hook():
    from amanda.conversion.tf import FuncSessionHook

    def begin():
        nonlocal context
        context = disabled()
        context.__enter__()

    def after_create_session(session, coord):
        context.__exit__(None, None, None)

    context = None
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

        if is_enabled():
            tf_graph = tf.get_default_graph()
            input_ops = {op.name for op in tf_graph.get_operations()}
            filter_hook = FilterHook()
            handler = register_inst_scope_hook(filter_hook)
            spec = model_fn(features, **kwargs)
            handler.unregister()
            forward_ops = {op.name for op in tf_graph.get_operations()}
            forward_ops = forward_ops - input_ops - filter_hook.disabled_ops
            tf_graph._forward_ops = forward_ops
        else:
            spec = model_fn(features, **kwargs)
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


def get_handler(wrapper):
    def handler(func, *args, **kwargs):
        return wrapper(func)(*args, **kwargs)

    return handler


def register_import_hook() -> None:
    import intercepts
    import tensorflow as tf

    intercepts.register(tf.Session.run, get_handler(session_run_wrapper))
    intercepts.register(tf.estimator.Estimator.train, get_handler(train_wrapper))
    intercepts.register(tf.estimator.Estimator.predict, get_handler(predict_wrapper))
    intercepts.register(tf.estimator.Estimator.evaluate, get_handler(evaluate_wrapper))

    # register_updater(
    #     MethodUpdater(
    #         module="tensorflow.python.client.session",
    #         cls="Session",
    #         method="run",
    #         decorator=session_run_wrapper,
    #     )
    # )
    # register_updater(
    #     MethodUpdater(
    #         module="tensorflow_estimator.python.estimator.estimator",
    #         cls="Estimator",
    #         method="train",
    #         decorator=train_wrapper,
    #     )
    # )
    # register_updater(
    #     MethodUpdater(
    #         module="tensorflow_estimator.python.estimator.estimator",
    #         cls="Estimator",
    #         method="predict",
    #         decorator=predict_wrapper,
    #     )
    # )
    # register_updater(
    #     MethodUpdater(
    #         module="tensorflow_estimator.python.estimator.estimator",
    #         cls="Estimator",
    #         method="evaluate",
    #         decorator=evaluate_wrapper,
    #     )
    # )
