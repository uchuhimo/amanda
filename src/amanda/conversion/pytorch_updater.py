import functools
import inspect
import types
from collections import defaultdict
from functools import wraps
from threading import RLock
from typing import Any, Dict, List, MutableMapping, Set

from loguru import logger

from amanda.cache import is_cache_enabled
from amanda.conversion.amanda_torch_pybind import (
    HookRegisterer,
    amanda_add_pre_hook,
    amanda_remove_pre_hook,
)
from amanda.event import (
    Action,
    OpContext,
    after_backward_op_call,
    after_op_call,
    on_backward_op_call,
    on_op_call,
)
from amanda.import_hook import (
    MatchedClassUpdater,
    MatchedFunctionUpdater,
    MethodUpdater,
    Updater,
    check_enabled,
    disabled,
)
from amanda.lang import get_superclasses
from amanda.tool import get_apply_scope, get_tools, register_cleanup_task

_lock: RLock = RLock()
_apply_scope = None
_grad_ids: MutableMapping[Any, Dict[int, int]] = {}
_cached_actions: Dict[int, Dict[str, List[Action]]] = {}
_has_cached_pre_actions: Dict[int, bool] = defaultdict(lambda: True)
_has_cached_post_actions: Dict[int, bool] = defaultdict(lambda: True)
_debug_cache: bool = False
_should_hit = False


def cleanup():
    global _apply_scope
    _apply_scope = None
    with _lock:
        _grad_ids.clear()
        _cached_actions.clear()
        _has_cached_pre_actions.clear()
        _has_cached_post_actions.clear()


def apply_insert_before_op(action, inputs):
    import torch

    # logger.debug("apply_insert_before_op")
    with torch.no_grad():
        filtered_inputs = inputs
        if action.inputs is not None:
            filtered_inputs = [filtered_inputs[index] for index in action.inputs]
        new_inputs = action.func(*filtered_inputs, **action.kwargs)
        if new_inputs is not None:
            if isinstance(new_inputs, torch.Tensor):
                new_inputs = [new_inputs]
            for index, new_input in zip(
                action.inputs or range(len(inputs)), new_inputs
            ):
                inputs[index] = new_input


def apply_insert_after_op(action, outputs):
    import torch

    # logger.debug("apply_insert_after_op")
    with torch.no_grad():
        filtered_outputs = outputs
        if action.outputs is not None:
            filtered_outputs = [filtered_outputs[index] for index in action.outputs]
        new_outputs = action.func(*filtered_outputs, **action.kwargs)
        if new_outputs is not None:
            if isinstance(new_outputs, torch.Tensor):
                new_outputs = [new_outputs]
            for index, new_output in zip(
                action.outputs or range(len(outputs)), new_outputs
            ):
                outputs[index] = new_output


def apply_replace_op(action, inputs):
    import torch

    logger.debug("apply_replace_op")
    with torch.no_grad():
        filtered_inputs = inputs
        if action.inputs is not None:
            filtered_inputs = [filtered_inputs[index] for index in action.inputs]
        return action.func(*filtered_inputs, **action.kwargs)


def next_id(id):
    # return (id * 1103515245 + 12345) & 0x7FFFFFFF
    return (id * 25214903917 + 11) & 0xFFFF_FFFF_FFFF


_grad_fns: Set[Any] = set()


class Ref:
    def __init__(self, ref):
        self.ref = ref


def register_bw_events_recursively(context, outputs, is_cached):
    """
    same functionality as register_bw_events() with subgraph matching,
    in this manner, a EventContext in bw phase have "op", "bw_op" two context,
    either of them may be None or not exists, denoting only exists in fw or bw,
    """
    import torch

    def _register_bw_events(context, grad_fn):
        def before_bw_op_hook(grad_output, context, bw_op, handle):
            with disabled(), _lock:
                if isinstance(grad_output, torch.Tensor):
                    grad_outputs = [grad_output]
                else:
                    grad_outputs = list(grad_output)
                bw_raw_op = bw_op.__class__
                self_id = id(bw_raw_op)
                # print(f"bw input of {id(bw_op)}")
                is_unknown = Ref(True)
                next_seed = Ref(self_id)
                if _debug_cache:
                    bw_op_id, cached_inputs = calc_op_id(
                        self_id,
                        grad_outputs,
                        bw_op=bw_op,
                        next_seed=next_seed,
                        is_unknown=is_unknown,
                    )
                    _cache_tracer.op_to_inputs[id(bw_op)] = cached_inputs
                    _cache_tracer.id_to_cache[id(bw_op)] = [
                        "bw_op",
                        bw_op,
                        "sid:",
                        bw_op_id,
                    ]
                else:
                    bw_op_id = calc_op_id(
                        self_id,
                        grad_outputs,
                        bw_op=bw_op,
                        next_seed=next_seed,
                        is_unknown=is_unknown,
                    )
                context["is_unknown"] = is_unknown

                is_cached = False
                if is_cache_enabled() and bw_op_id in _cached_actions:
                    # print("hit", bw_raw_op, bw_op_id)
                    cached_actions = _cached_actions[bw_op_id]
                    is_cached = True
                else:
                    cached_actions = {
                        "insert_before_backward_op": [],
                    }
                    if _debug_cache and _should_hit:
                        print()
                        _cache_tracer.print_trace_from_op(id(bw_op))
                        raise RuntimeError()
                        # pass

                if is_cached:
                    for action in cached_actions["insert_before_backward_op"]:
                        apply_insert_before_op(action, grad_outputs)
                else:
                    with torch.no_grad():
                        context.trigger(
                            on_backward_op_call,
                            backward_op=bw_raw_op,
                            backward_op_id=bw_op_id,
                            grad_outputs=grad_outputs,
                        )
                    for action in list(context.actions):
                        if action.type == "insert_before_backward_op":
                            apply_insert_before_op(action, grad_outputs)
                            context.actions.remove(action)
                            cached_actions["insert_before_backward_op"].append(action)

                if is_cached:
                    context.update(
                        backward_op=bw_raw_op,
                        backward_op_id=bw_op_id,
                        grad_outputs=grad_outputs,
                    )
                elif is_cache_enabled():
                    _cached_actions[bw_op_id] = cached_actions
                    if len(cached_actions["insert_before_backward_op"]) == 0:
                        _has_cached_pre_actions[bw_subgraph_id] = False

                assert amanda_remove_pre_hook(bw_op, handle)
                return tuple(grad_outputs)

        def after_bw_op_hook(
            grad_input, grad_output, context, bw_op, bw_subgraph_id, handle
        ):
            with disabled(), _lock:
                _grad_fns.remove(bw_op)
                if isinstance(grad_input, torch.Tensor):
                    grad_inputs = [grad_input]
                else:
                    grad_inputs = grad_input
                new_grad_inputs = list(grad_inputs)

                bw_op_id = context.backward_op_id
                # print(f"bw_op: {bw_op.__class__}, bw_op_id: {bw_op_id}")
                is_cached = False
                cached_actions = _cached_actions.get(bw_op_id, {})
                if (
                    is_cache_enabled()
                    and "insert_after_backward_op" in _cached_actions[bw_op_id]
                ):
                    is_cached = True
                else:
                    cached_actions["replace_backward_op"] = []
                    cached_actions["insert_after_backward_op"] = []

                if is_cached:
                    for action in cached_actions["replace_backward_op"]:
                        new_input = apply_replace_op(action, context.grad_outputs)
                        if isinstance(new_input, torch.Tensor):
                            new_grad_inputs = [new_input]
                        else:
                            new_grad_inputs = new_input
                        break
                else:
                    with torch.no_grad():
                        context.trigger(
                            after_backward_op_call,
                            grad_inputs=new_grad_inputs,
                        )
                    for action in list(context.actions):
                        if action.type == "replace_op":
                            new_input = apply_replace_op(action, context.grad_outputs)
                            context.actions.remove(action)
                            cached_actions["replace_backward_op"].append(action)
                            if isinstance(new_input, torch.Tensor):
                                new_grad_inputs = [new_input]
                            else:
                                new_grad_inputs = new_input
                            break

                if is_cached:
                    for action in cached_actions["insert_after_backward_op"]:
                        apply_insert_after_op(action, new_grad_inputs)
                else:
                    for action in list(context.actions):
                        if action.type == "insert_after_backward_op":
                            apply_insert_after_op(action, new_grad_inputs)
                            context.actions.remove(action)
                            cached_actions["insert_after_backward_op"].append(action)
                    assert len(context.actions) == 0

                if not is_cached and is_cache_enabled():
                    if (
                        len(cached_actions["replace_backward_op"]) == 0
                        and len(cached_actions["insert_after_backward_op"]) == 0
                    ):
                        _has_cached_post_actions[bw_subgraph_id] = False
                        if len(cached_actions["insert_before_backward_op"]) == 0:
                            _cached_actions[context.op_id][
                                "has_backward_actions"
                            ] = False
                arg_id = next_id(bw_op_id)
                if _debug_cache:
                    for grad_input, (next_op, input_index) in zip(
                        grad_inputs, bw_op.next_functions
                    ):
                        if grad_input is not None:
                            if id(next_op) not in _grad_ids:
                                _grad_ids[id(next_op)] = {}
                            if context["is_unknown"].ref:
                                _grad_ids[id(next_op)].pop(input_index, None)
                            else:
                                _cache_tracer.output_to_op[
                                    (id(next_op), input_index)
                                ] = id(bw_op)
                                _cache_tracer.id_to_cache[
                                    (id(next_op), input_index)
                                ] = [
                                    "bw_output",
                                    type(grad_input),
                                    "id:",
                                    id(grad_input),
                                    "sid:",
                                    arg_id,
                                ]
                                _grad_ids[id(next_op)][input_index] = arg_id
                        arg_id = next_id(arg_id)
                else:
                    for grad_input, (next_op, input_index) in zip(
                        grad_inputs, bw_op.next_functions
                    ):
                        if grad_input is not None:
                            if id(next_op) not in _grad_ids:
                                _grad_ids[id(next_op)] = {}
                            if context["is_unknown"].ref:
                                _grad_ids[id(next_op)].pop(input_index, None)
                            else:
                                _grad_ids[id(next_op)][input_index] = arg_id
                        arg_id = next_id(arg_id)
                for grad_input, new_grad_input in zip(grad_inputs, new_grad_inputs):
                    if isinstance(grad_input, torch.Tensor) and isinstance(
                        new_grad_input, torch.Tensor
                    ):
                        grad_input.data = new_grad_input
                handle.remove()

        if not (grad_fn and grad_fn not in _grad_fns):
            return
        _grad_fns.add(grad_fn)
        bw_context = context.inherite()
        nonlocal bw_subgraph_id
        local_bw_subgraph_id = bw_subgraph_id
        if not is_cached or _has_cached_pre_actions[local_bw_subgraph_id]:
            # if _debug_cache and _should_hit:
            #     print("before_bw_op_hook", grad_fn)
            pre_handle = amanda_add_pre_hook(
                grad_fn,
                lambda grad_output: before_bw_op_hook(
                    grad_output,
                    context=bw_context,
                    bw_op=grad_fn,
                    handle=pre_handle,
                ),
            )
        if not is_cached or _has_cached_post_actions[local_bw_subgraph_id]:
            # if _debug_cache and _should_hit:
            #     print("after_bw_op_hook", grad_fn)
            post_handle = grad_fn.register_hook(
                lambda grad_input, grad_output: after_bw_op_hook(
                    grad_input,
                    grad_output,
                    context=bw_context,
                    bw_op=grad_fn,
                    bw_subgraph_id=local_bw_subgraph_id,
                    handle=post_handle,
                )
            )
        for next_grad_fn, next_input_pos in grad_fn.next_functions:
            # print("x3", bw_subgraph_id.ref)
            bw_subgraph_id = next_id(bw_subgraph_id)
            # print("x4", bw_subgraph_id.ref)
            _register_bw_events(context, next_grad_fn)

    bw_subgraph_id = context.op_id
    # print(f"begin {context.op}")
    for output in outputs:
        bw_subgraph_id = next_id(bw_subgraph_id)
        if hasattr(output, "grad_fn") and output.requires_grad:
            # if _debug_cache and _should_hit:
            #     print("_register_bw_events", context.op, type(output))
            # print(f"is_cached: {is_cached}, bw_subgraph_id: {bw_subgraph_id}")
            _register_bw_events(context, output.grad_fn)
        # print("x1", bw_subgraph_id.ref)
        # print("x2", bw_subgraph_id.ref)
    # print("end")


""" unpack_input_grad_fns()
unpack a nested iterable (currently list and set supported)
    of tensors recursively into a list,
the grad_fn of each tensor is gathered into the output list,
this list is used to identify the backward subgraph tracing boundary
"""


def unpack_input_grad_fns(inputs):
    def _unpack_input_grad_fns(inputs):
        for input in inputs:
            if hasattr(input, "grad_fn") and input.grad_fn:
                input_grad_fns.append(input.grad_fn)
            elif isinstance(input, list) or isinstance(input, set):
                _unpack_input_grad_fns(input)

    input_grad_fns = []
    _unpack_input_grad_fns(inputs)
    return input_grad_fns


class CacheTracer:
    def __init__(self):
        self.op_to_inputs = {}
        self.output_to_op = {}
        self.id_to_cache = {}

    def print_trace_from_op(self, id, indent=0):
        print(" " * indent + str(id) + str(self.id_to_cache[id]))
        if indent > 10:
            print(" " * (indent + 2) + "...")
            return
        for input in self.op_to_inputs[id]:
            self.print_trace_from_input(input, indent=indent + 2)

    def print_trace_from_input(self, id, indent=0):
        print(" " * indent + str(id) + str(self.id_to_cache[id]))
        if indent > 10:
            print(" " * (indent + 2) + "...")
            return
        if id in self.output_to_op:
            self.print_trace_from_op(self.output_to_op[id], indent=indent + 2)


_cache_tracer: CacheTracer = CacheTracer()


def get_cache_size():
    return len(_cached_actions)


def get_input_id(input, default_id, next_seed=None, inputs=None, is_unknown=None):
    import torch

    if _debug_cache:
        if isinstance(input, torch.nn.Parameter):
            inputs.append(id(input))
            _cache_tracer.id_to_cache[id(input)] = ["parameter", type(input)]
            is_unknown.ref = False
            return id(input)
        elif hasattr(input, "__stable_id__"):
            inputs.append(id(input))
            if hasattr(input, "__is__buffer__"):
                _cache_tracer.id_to_cache[id(input)] = [
                    "buffer",
                    type(input),
                    "sid:",
                    input.__stable_id__,
                ]
            elif hasattr(input, "__is__state__"):
                _cache_tracer.id_to_cache[id(input)] = [
                    "state",
                    type(input),
                    "sid:",
                    input.__stable_id__,
                ]
            elif hasattr(input, "__is__grad__"):
                _cache_tracer.id_to_cache[id(input)] = [
                    "grad",
                    type(input),
                    "sid:",
                    input.__stable_id__,
                ]
            is_unknown.ref = False
            return input.__stable_id__
        elif isinstance(input, (list, tuple)):
            input_id = default_id
            for x in input:
                next_seed.ref = next_id(next_seed.ref)
                input_id = input_id ^ get_input_id(
                    x,
                    default_id=next_seed.ref,
                    next_seed=next_seed,
                    inputs=inputs,
                    is_unknown=is_unknown,
                )
            return input_id
        else:
            inputs.append(id(input))
            _cache_tracer.id_to_cache[id(input)] = [
                "unknown",
                type(input),
                "default_id:",
                default_id,
            ]
            return default_id
    else:
        if isinstance(input, torch.nn.Parameter):
            is_unknown.ref = False
            return id(input)
        elif hasattr(input, "__stable_id__"):
            is_unknown.ref = False
            return input.__stable_id__
        elif isinstance(input, (list, tuple)):
            input_id = default_id
            for x in input:
                next_seed.ref = next_id(next_seed.ref)
                input_id = input_id ^ get_input_id(
                    x,
                    default_id=next_seed.ref,
                    next_seed=next_seed,
                    is_unknown=is_unknown,
                )
            return input_id
        else:
            return default_id


def calc_op_id(self_id, args, kwargs=None, bw_op=None, next_seed=None, is_unknown=None):
    if _debug_cache:
        inputs = []
        op_id = 0
        for index, input in enumerate(args):
            next_seed.ref = next_id(next_seed.ref)
            if (
                bw_op is not None
                and id(bw_op) in _grad_ids
                and index in _grad_ids[id(bw_op)]
            ):
                inputs.append((id(bw_op), index))
                op_id = op_id ^ _grad_ids[id(bw_op)][index]
                is_unknown.ref = False
            else:
                op_id = op_id ^ get_input_id(
                    input,
                    default_id=next_seed.ref,
                    next_seed=next_seed,
                    inputs=inputs,
                    is_unknown=is_unknown,
                )
        if kwargs is not None:
            for name, input in kwargs.items():
                op_id = op_id ^ get_input_id(
                    input,
                    default_id=id(name),
                    next_seed=next_seed,
                    inputs=inputs,
                    is_unknown=is_unknown,
                )
        op_id = op_id ^ self_id
        return op_id, inputs
    else:
        op_id = 0
        for index, input in enumerate(args):
            next_seed.ref = next_id(next_seed.ref)
            if (
                bw_op is not None
                and id(bw_op) in _grad_ids
                and index in _grad_ids[id(bw_op)]
            ):
                op_id = op_id ^ _grad_ids[id(bw_op)][index]
                is_unknown.ref = False
            else:
                op_id = op_id ^ get_input_id(
                    input,
                    default_id=next_seed.ref,
                    next_seed=next_seed,
                    is_unknown=is_unknown,
                )
        if kwargs is not None:
            for name, input in kwargs.items():
                op_id = op_id ^ get_input_id(
                    input,
                    default_id=id(name),
                    next_seed=next_seed,
                    is_unknown=is_unknown,
                )
        op_id = op_id ^ self_id
        return op_id


def function_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import torch

        with disabled(), _lock:
            global _apply_scope
            if _apply_scope is not None and _apply_scope != get_apply_scope():
                assert _apply_scope == get_apply_scope()
                _apply_scope = None
                _cached_actions.clear()
                _has_cached_pre_actions.clear()
                _has_cached_post_actions.clear()
            if _apply_scope is None:
                _apply_scope = get_apply_scope()
                register_cleanup_task(cleanup)
            tools = get_tools()
            inputs = [*args, kwargs]
            raw_func = func
            if isinstance(func, functools.partial):
                raw_func = func.__wrapped__
            context = OpContext(tools=tools, namespace="pytorch")
            raw_op = raw_func
            if isinstance(raw_func, types.MethodWrapperType):
                raw_op = raw_func.__self__
            self_id = id(raw_op)
            is_unknown = Ref(True)
            next_seed = Ref(self_id)
            if _debug_cache:
                op_id, cached_inputs = calc_op_id(
                    self_id, args, kwargs, next_seed=next_seed, is_unknown=is_unknown
                )
                _cache_tracer.op_to_inputs[op_id] = cached_inputs
                _cache_tracer.id_to_cache[op_id] = ["op", raw_op]
            else:
                op_id = calc_op_id(
                    self_id, args, kwargs, next_seed=next_seed, is_unknown=is_unknown
                )
            # print(f"op: {raw_func}, op_id: {op_id}")

            is_cached = False
            if is_cache_enabled() and op_id in _cached_actions:
                # print("hit", raw_func.__name__, op_id)
                cached_actions = _cached_actions[op_id]
                is_cached = True
            else:
                cached_actions = {
                    "insert_before_op": [],
                    "replace_op": [],
                    "insert_after_op": [],
                    "has_backward_actions": True,
                }
                if _debug_cache and _should_hit:
                    # print("miss", raw_func.__name__, op_id)
                    print()
                    _cache_tracer.print_trace_from_op(op_id)
                    raise RuntimeError()
                    # pass
            # if _debug_cache and _should_hit and raw_op.__name__ == "grad":
            # if _debug_cache and raw_op.__name__ == "grad":
            #     print()
            #     _cache_tracer.print_trace_from_op(op_id)

            if is_cached:
                for action in cached_actions["insert_before_op"]:
                    apply_insert_before_op(action, inputs)
            else:
                with torch.no_grad():
                    context.trigger(
                        on_op_call,
                        op=raw_func,
                        op_id=op_id,
                        inputs=inputs,
                    )
                for action in list(context.actions):
                    if action.type == "insert_before_op":
                        apply_insert_before_op(action, inputs)
                        context.actions.remove(action)
                        cached_actions["insert_before_op"].append(action)

            is_replaced = False

            if is_cached:
                for action in cached_actions["replace_op"]:
                    output = apply_replace_op(action, inputs)
                    is_replaced = True
                    break
            else:
                for action in list(context.actions):
                    if action.type == "replace_op":
                        output = apply_replace_op(action, inputs)
                        context.actions.remove(action)
                        cached_actions["replace_op"].append(action)
                        is_replaced = True
                        break

            if not is_replaced:
                output = func(*inputs[:-1], **inputs[-1])
            if type(output) != tuple:
                outputs = [output]
                is_output_nested = True
            else:
                outputs = output
                is_output_nested = False

            if is_cached:
                for action in cached_actions["insert_after_op"]:
                    apply_insert_after_op(action, outputs)
            else:
                with torch.no_grad():
                    context.trigger(
                        after_op_call,
                        outputs=outputs,
                    )
                for action in list(context.actions):
                    if action.type == "insert_after_op":
                        apply_insert_after_op(action, outputs)
                        context.actions.remove(action)
                        cached_actions["insert_after_op"].append(action)
                assert len(context.actions) == 0

            if is_cached:
                context.update(
                    op=raw_func,
                    op_id=op_id,
                    inputs=inputs,
                    outputs=outputs,
                )
            elif is_cache_enabled():
                _cached_actions[op_id] = cached_actions
            arg_id = op_id
            if (not is_unknown.ref) and (
                (not raw_op.__name__.endswith("_")) or raw_op.__name__.endswith("__")
            ):
                if _debug_cache:
                    for output in outputs:
                        arg_id = next_id(arg_id)
                        if (
                            isinstance(output, torch.Tensor)
                            and not isinstance(output, torch.nn.Parameter)
                            and not hasattr(output, "__stable_id__")
                        ):
                            _cache_tracer.output_to_op[id(output)] = op_id
                            _cache_tracer.id_to_cache[id(output)] = [
                                "output",
                                type(output),
                                "sid:",
                                arg_id,
                            ]
                            output.__stable_id__ = arg_id
                else:
                    for output in outputs:
                        arg_id = next_id(arg_id)
                        if (
                            isinstance(output, torch.Tensor)
                            and not isinstance(output, torch.nn.Parameter)
                            and not hasattr(output, "__stable_id__")
                        ):
                            output.__stable_id__ = arg_id
            if torch.is_grad_enabled() and (
                not is_cached or cached_actions["has_backward_actions"]
            ):
                register_bw_events_recursively(context, outputs, is_cached)
            if is_output_nested:
                output = outputs[0]
            else:
                output = outputs
            return output

    return check_enabled(func, wrapper)


def register_buffer_wrapper(func):
    @wraps(func)
    def wrapper(self, name, tensor, persistent=True):
        if tensor is not None:
            tensor.__stable_id__ = id(tensor)
            tensor.__is_buffer__ = True
        return func(self, name, tensor, persistent)

    return wrapper


def _hook_for_profile_wrapper(_hook_for_profile_func):
    @wraps(_hook_for_profile_func)
    def wrapper(self):
        def hook_step(step_func):
            import torch

            @wraps(step_func)
            def wrapper(self, closure=None):
                step_func(self, closure)
                for group in self.param_groups:
                    for p in group["params"]:
                        state = self.state[p]
                        for value in state.values():
                            if isinstance(value, torch.Tensor):
                                value.__stable_id__ = id(value)
                                value.__is_state__ = True

            return wrapper

        hooked = getattr(self.__class__.step, "hooked_by_amanda", None)
        if not hooked:
            self.__class__.step = hook_step(self.__class__.step)
            self.__class__.step.hooked_by_amanda = True
        return _hook_for_profile_func(self)

    return wrapper


def step_wrapper(func):
    import torch

    @wraps(func)
    def wrapper(self, closure=None):
        func(self, closure)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.__stable_id__ = id(value)
                        value.__is_state__ = True

    return wrapper


def set_grad_wrapper(func):
    import torch

    @wraps(func)
    def wrapper(self, grad):
        if grad is not None:
            if isinstance(self, torch.nn.Parameter):
                grad.__stable_id__ = next_id(id(self))
                grad.__is_grad__ = True
            elif hasattr(self, "__is_buffer__"):
                grad.__stable_id__ = next_id(self.__stable_id__)
                grad.__is_grad__ = True
            elif hasattr(self, "__is_state__"):
                grad.__stable_id__ = next_id(self.__stable_id__)
                grad.__is_grad__ = True
        return func(self, grad)

    return wrapper


def get_grad_wrapper(func):
    import torch

    @wraps(func)
    def wrapper(self):
        grad = func(self)
        if grad is not None:
            if isinstance(self, torch.nn.Parameter):
                grad.__stable_id__ = next_id(id(self))
                grad.__is_grad__ = True
            elif hasattr(self, "__is_buffer__"):
                grad.__stable_id__ = next_id(self.__stable_id__)
                grad.__is_grad__ = True
            elif hasattr(self, "__is_state__"):
                grad.__stable_id__ = next_id(self.__stable_id__)
                grad.__is_grad__ = True
        return grad

    return wrapper


class ModuleUpdater(MatchedClassUpdater):
    def __init__(self):
        super().__init__(module="", method="forward", decorator=function_wrapper)

    def is_match(self, name: str) -> bool:
        return True

    def is_match_class(self, module: str, name: str, cls) -> bool:
        superclasses = get_superclasses(cls)
        return (
            "torch.nn.modules.module.Module" in superclasses
            and "torch.jit.ScriptModule" not in superclasses
        )


class FunctionalUpdater(MatchedFunctionUpdater):
    def __init__(self, modules: List[str]):
        super().__init__(module="", decorator=function_wrapper)
        self.modules = modules

    def is_match(self, name: str) -> bool:
        return name in self.modules

    def is_match_func(self, module: str, name: str, func) -> bool:
        if name in ["conv2d", "_max_pool2d"]:
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(module, name)
        boolean_dispatch_functions = [
            "fractional_max_pool2d",
            "fractional_max_pool3d",
            "max_pool1d",
            "max_pool2d",
            "max_pool3d",
            "adaptive_max_pool1d",
            "adaptive_max_pool2d",
            "adaptive_max_pool3d",
        ]
        if name in boolean_dispatch_functions:
            return False
        if name.startswith("_") and name[1:] in boolean_dispatch_functions:
            func.op_type = name[1:]
            return True
        return not name.startswith("_")


TORCH_OP_LIST: Set[str] = set()

TORCH_OP_OVERLOAD_LIST = (
    "__add__",
    "__radd__",
    "__iadd__",
    "__rmul__",
    "__mul__",
    "__imul__",
    "__sub__",
    "__isub__",
    "__div__",
    "__truediv__",
    "__floordiv__",
    "__idiv__",
    "__ifloordiv__",
    "__mod__",
    "__imod__",
    "__invert__",
    "__matmul__",
    "__and__",
    "__iand__",
    "__ilshift__",
    "__ixor__",
    "__ior__",
    "__irshift__",
    "__lshift__",
    "__or__",
    "__rshift__",
    "__xor__",
)

TORCH_OP_LIST.update(TORCH_OP_OVERLOAD_LIST)


def wrap_op(module, name, wrapper):
    import torch

    from amanda import intercepts

    handler = intercepts.to_handler(wrapper)
    if isinstance(module, torch._C._VariableFunctionsClass):
        if hasattr(module, name):
            intercepts.register(getattr(module, name), handler, "amanda")
            return True
    elif name in module.__dict__:
        intercepts.register(module.__dict__[name], handler, "amanda")
        return True
    return False


def listener_callback(op_name: str) -> str:
    def remove_namespace(name: str) -> str:
        pos = name.find("::")
        if not pos == -1:
            return name[pos + 2 :]
        else:
            return name

    import torch

    name = remove_namespace(op_name)
    # if name in ["is_leaf"]:
    #     return op_name
    for module in [
        torch._C._nn,
        torch._C._fft,
        torch._C._linalg,
        torch._C._TensorBase,
        torch._C._VariableFunctions,
    ]:
        wrap_op(module, name, function_wrapper)
    return op_name


class ListenerFunctionalUpdater(Updater):
    def __init__(self, module: str, submodules: List[str]):
        self.module = module
        self.submodules = set(submodules)
        self.decorator = function_wrapper

    def is_match(self, name: str) -> bool:
        return name == self.module

    def update_module(self, module, submodule_name, func_name):
        func = getattr(module.__dict__[submodule_name], func_name)
        if hasattr(func, "updated") and func.updated:
            return
        new_func = self.decorator(func)
        new_func.original = func
        new_func.updated = True
        module.__dict__[submodule_name].__dict__[func_name] = new_func

    def update_class(self, module, submodule_name, func_name):
        func = getattr(module.__dict__[submodule_name], func_name)
        if hasattr(func, "updated") and func.updated:
            return
        new_func = self.decorator(func)
        new_func.original = func
        new_func.updated = True
        setattr(module.__dict__[submodule_name], func_name, new_func)

    def update_object(self, module, submodule_name, submodule_cls_name, func_name):
        func = getattr(module.__dict__[submodule_cls_name], func_name)
        if hasattr(func, "updated") and func.updated:
            return
        new_func = self.decorator(func)
        new_func.original = func
        new_func.updated = True
        setattr(module.__dict__[submodule_cls_name], func_name, new_func)
        setattr(module.__dict__[submodule_name], func_name, new_func)

    def add_docstr_wrapper(self, _add_docstr):
        @wraps(_add_docstr)
        def wrapper(func, doc_str):
            if hasattr(func, "updated") and func.updated:
                func.__dict__["original"] = _add_docstr(
                    getattr(func, "original"), doc_str
                )
                return func
            else:
                return _add_docstr(func, doc_str)

        return wrapper

    def update(self, module) -> None:

        global TORCH_OP_LIST

        submodules = dict(module.__dict__)
        for submodule_key in submodules:
            if submodule_key == "_add_docstr":
                module.__dict__[submodule_key] = self.add_docstr_wrapper(
                    module.__dict__[submodule_key]
                )
                continue

            if submodule_key in self.submodules:

                if (
                    not inspect.ismodule(module.__dict__[submodule_key])
                    and type(module.__dict__[submodule_key]) == type
                ):
                    module.__dict__[submodule_key] = type(
                        submodule_key,
                        (module.__dict__[submodule_key],),
                        dict(module.__dict__[submodule_key].__dict__),
                    )
                    funcs = dict(module.__dict__[submodule_key].__dict__)
                    for func_key in funcs:
                        if func_key not in TORCH_OP_LIST:
                            continue
                        if func_key == "data":
                            print(f'skip "data" of {submodule_key}')
                            continue
                        self.update_class(module, submodule_key, func_key)
                elif (
                    not inspect.ismodule(module.__dict__[submodule_key])
                    and type(module.__dict__[submodule_key]) != type
                ):
                    submodule_cls_key = module.__dict__[
                        submodule_key
                    ].__class__.__name__
                    module.__dict__[submodule_cls_key] = type(
                        submodule_cls_key,
                        (object,),
                        dict(module.__dict__[submodule_cls_key].__dict__),
                    )
                    funcs = dict(module.__dict__[submodule_cls_key].__dict__)
                    module.__dict__[submodule_key] = module.__dict__[
                        submodule_cls_key
                    ]()
                    for func_key in funcs:
                        if func_key not in TORCH_OP_LIST:
                            continue
                        # print(module.__name__, submodule_cls_key, func_key)
                        self.update_object(
                            module, submodule_key, submodule_cls_key, func_key
                        )
                else:
                    funcs = dict(module.__dict__[submodule_key].__dict__)
                    for func_key in funcs:
                        if func_key not in TORCH_OP_LIST:
                            continue
                        # print(module.__name__, submodule_key, func_key)
                        self.update_module(module, submodule_key, func_key)


def grad_fn_wrapper(grad_fn):
    def getter_wrapper(getter):
        @wraps(getter)
        def wrapper(self):
            fn = getter(self)
            if fn is None:
                return None
            else:
                print("getter")
                return fn_wrapper(fn)

        return wrapper

    def fn_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            print("grad_fn")
            if hasattr(self, "_grad_fn_decorator"):
                return self._grad_fn_decorator(func)(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return property(
        getter_wrapper(grad_fn.__get__), grad_fn.__set__, grad_fn.__delete__
    )


class GradFnUpdater(MethodUpdater):
    def __init__(self):
        super().__init__(
            module="torch.tensor",
            cls="Tensor",
            method="grad_fn",
            decorator=grad_fn_wrapper,
        )


def register_import_hook() -> None:
    pass


def register_intercepts() -> None:
    import torch

    from amanda import intercepts

    HookRegisterer(listener_callback)
    intercepts.register(
        torch.nn.Module.register_buffer, intercepts.to_handler(register_buffer_wrapper)
    )
    # intercepts.register(
    #     torch.optim.Optimizer.step, intercepts.to_handler(step_wrapper)
    # )
    intercepts.register(
        torch.optim.Optimizer._hook_for_profile,
        intercepts.to_handler(_hook_for_profile_wrapper),
    )
    intercepts.register(torch.Tensor.grad.fset, intercepts.to_handler(set_grad_wrapper))
    intercepts.register(torch.Tensor.grad.fget, intercepts.to_handler(get_grad_wrapper))
