import functools
import inspect
import weakref
from functools import wraps
from typing import Any, List, MutableMapping, Set

from loguru import logger

from amanda.conversion.amanda_torch_pybind import (
    HookRegisterer,
    amanda_add_pre_hook,
    amanda_remove_pre_hook,
)
from amanda.event import (
    OpContext,
    after_backward_op_call,
    after_backward_op_executed,
    after_op_call,
    before_backward_op_executed,
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
from amanda.tool import get_tools


def registry_bw_events(context, output):
    """
    registry the backward hooks for before/after backward op events.
    before_bw_op_event is hooked on output tensor.
    after_bw_op_event is hooked on backward function of output tensor.
        it is the output.grad_fn.
    note that this method is coupled with pytorch and need to be refactored

    hook wrapper= triggers the event with context and remove the hook.
    note that if a hook is not triggered, then it will not be collected!
    this may caused by ops only existed only in forward phase.
    """

    def before_bw_op_hook(context, output):
        context.trigger(before_backward_op_executed, output_grad=output)
        before_bw_op_hook_handle.remove()

    def after_bw_op_hook(context, input, output):
        context.trigger(
            after_backward_op_executed, input_grad=input, output_grad=output
        )
        after_bw_op_hook_handle.remove()

    if hasattr(output, "register_hook") and output.requires_grad:
        before_bw_op_hook_handle = output.register_hook(
            lambda output: before_bw_op_hook(context, output)
        )

    if hasattr(output, "grad_fn"):  # skip ops with non-tensor output
        if (
            output.grad_fn.__class__.__name__ == "UnsafeViewBackward"
        ):  # check for broadcast case
            """check for broadcast case
            not considering broadcast of input tensors now
            """
            # print(f"add backward with broadcast")
            if output.grad_fn.next_functions[0][0]:
                after_bw_op_hook_handle = output.grad_fn.next_functions[0][
                    0
                ].register_hook(
                    lambda input, output: after_bw_op_hook(
                        context, input, context["output_grad"]
                    )
                )
        elif output.grad_fn.__class__.__name__ == "AddBackward0":  # check for bias
            # print(f"add backward with bias founded")
            if output.grad_fn.next_functions[0][0]:
                after_bw_op_hook_handle = output.grad_fn.next_functions[0][
                    0
                ].register_hook(
                    lambda input, output: after_bw_op_hook(
                        context, input, context["output_grad"]
                    )
                )
        else:
            if output.grad_fn:
                after_bw_op_hook_handle = output.grad_fn.register_hook(
                    lambda input, output: after_bw_op_hook(context, input, output)
                )
    else:
        pass


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
    logger.debug("apply_replace_op")
    filtered_inputs = inputs
    if action.inputs is not None:
        filtered_inputs = [filtered_inputs[index] for index in action.inputs]
    return action.func(*filtered_inputs, **action.kwargs)


_grad_fns: Set[Any] = set()


def register_bw_events_recursively(context, outputs, input_grad_fns):
    """
    same functionality as register_bw_events() with subgraph matching,
    in this manner, a EventContext in bw phase have "op", "bw_op" two context,
    either of them may be None or not exists, denoting only exists in fw or bw,
    """
    import torch

    def _register_bw_events(context, grad_fn):
        def before_bw_op_hook(grad_output, context, bw_op, handle):
            with disabled():
                if isinstance(grad_output, torch.Tensor):
                    grad_outputs = [grad_output]
                else:
                    grad_outputs = grad_output
                bw_raw_op = bw_op.__class__
                bw_op_id = calc_op_id(bw_raw_op, grad_outputs)
                context.trigger(
                    on_backward_op_call,
                    backward_op=bw_raw_op,
                    backward_op_id=bw_op_id,
                    grad_outputs=grad_outputs,
                )
                new_grad_outputs = list(grad_outputs)
                for action in list(context.actions):
                    if action.type == "insert_before_backward_op":
                        apply_insert_before_op(action, new_grad_outputs)
                        context.actions.remove(action)
                assert amanda_remove_pre_hook(bw_op, handle)
                assert len(grad_outputs) == len(new_grad_outputs)
                return tuple(new_grad_outputs)

        def after_bw_op_hook(grad_input, grad_output, context, bw_op, handle):
            with disabled():
                _grad_fns.remove(bw_op)
                if isinstance(grad_input, torch.Tensor):
                    grad_inputs = [grad_input]
                else:
                    grad_inputs = grad_input
                if isinstance(grad_output, torch.Tensor):
                    grad_outputs = [grad_output]
                else:
                    grad_outputs = grad_output
                bw_raw_op = bw_op.__class__
                if "backward_op_id" not in context:
                    context["backward_op_id"] = calc_op_id(bw_raw_op, grad_outputs)
                context.trigger(
                    after_backward_op_call,
                    backward_op=bw_raw_op,
                    grad_outputs=grad_outputs,
                    grad_inputs=grad_inputs,
                )
                for action in list(context.actions):
                    if action.type == "replace_op":
                        new_input = apply_replace_op(action, grad_outputs)
                        context.actions.remove(action)
                        if isinstance(new_input, torch.Tensor):
                            grad_inputs = [new_input]
                        else:
                            grad_inputs = new_input
                        break
                new_grad_inputs = list(grad_inputs)
                for action in list(context.actions):
                    if action.type == "insert_after_backward_op":
                        apply_insert_after_op(action, new_grad_inputs)
                        context.actions.remove(action)
                assert len(context.actions) == 0
                for grad_input, new_grad_input in zip(grad_inputs, new_grad_inputs):
                    if isinstance(grad_input, torch.Tensor) and isinstance(
                        new_grad_input, torch.Tensor
                    ):
                        grad_input.data = new_grad_input
                handle.remove()

        if (
            grad_fn
            and grad_fn not in input_grad_fns
            and grad_fn not in registered_grad_fn
        ):
            _grad_fns.add(grad_fn)
            registered_grad_fn.append(grad_fn)
            bw_context = context.inherite()
            pre_handle = amanda_add_pre_hook(
                grad_fn,
                lambda grad_output: before_bw_op_hook(
                    grad_output,
                    context=bw_context,
                    bw_op=grad_fn,
                    handle=pre_handle,
                ),
            )
            post_handle = grad_fn.register_hook(
                lambda grad_input, grad_output: after_bw_op_hook(
                    grad_input,
                    grad_output,
                    context=bw_context,
                    bw_op=grad_fn,
                    handle=post_handle,
                )
            )
        else:
            return
        for next_grad_fn, next_input_pos in grad_fn.next_functions:
            if next_grad_fn and next_grad_fn not in input_grad_fns:
                _register_bw_events(context, next_grad_fn)

    registered_grad_fn = list()

    for output in outputs:
        if hasattr(output, "grad_fn"):
            _register_bw_events(context, output.grad_fn)


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


_tensor_ids: MutableMapping[Any, int] = weakref.WeakKeyDictionary()


def calc_op_id(op, args, kwargs=None):
    import torch

    op_id = 0
    arg_id = 1
    for input in args:
        if input in _tensor_ids:
            op_id = op_id ^ _tensor_ids[input]
        elif isinstance(input, torch.nn.Parameter):
            op_id = op_id ^ id(input)
        else:
            op_id = op_id ^ arg_id
        arg_id = arg_id << 1
    if kwargs is not None:
        for name, input in kwargs.items():
            if input in _tensor_ids:
                op_id = op_id ^ _tensor_ids[input]
            elif isinstance(input, torch.nn.Parameter):
                op_id = op_id ^ id(input)
            else:
                op_id = op_id ^ id(name)
    return op_id ^ id(op)


def function_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with disabled():
            tools = get_tools()
            input_grad_fns = unpack_input_grad_fns(args) + unpack_input_grad_fns(
                kwargs.values()
            )
            context = OpContext(tools=tools, namespace="pytorch")
            inputs = [*args, kwargs]
            raw_func = func
            if isinstance(func, functools.partial):
                raw_func = func.__wrapped__
            op_id = calc_op_id(raw_func, args, kwargs)
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
            is_replaced = False
            for action in list(context.actions):
                if action.type == "replace_op":
                    output = apply_replace_op(action, inputs)
                    context.actions.remove(action)
                    is_replaced = True
                    break
            if not is_replaced:
                output = func(*args, **kwargs)
            if type(output) != tuple:
                outputs = [output]
                is_output_nested = True
            else:
                outputs = output
                is_output_nested = False
            context.trigger(
                after_op_call,
                outputs=outputs,
            )
            for action in list(context.actions):
                if action.type == "insert_after_op":
                    apply_insert_after_op(action, outputs)
                    context.actions.remove(action)
            assert len(context.actions) == 0
            register_bw_events_recursively(context, outputs, input_grad_fns)
            if is_output_nested:
                output = outputs[0]
            else:
                output = outputs
            return output

    return check_enabled(func, wrapper)


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
    HookRegisterer(listener_callback)
