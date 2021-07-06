import inspect
from contextlib import contextmanager
from functools import wraps
from typing import List

from amanda.event import EventContext, after_op_executed, before_op_executed
from amanda.import_hook import (
    MatchedClassUpdater,
    MatchedFunctionUpdater,
    MethodUpdater,
    disabled,
    is_enabled,
    register_updater,
)
from amanda.lang import get_superclasses
from amanda.tool import Tool


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


def function_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_enabled():
            with disabled():

                if _tool is None:
                    return func(*args, **kwargs)
                input_grad_fns = unpack_input_grad_fns(args) + unpack_input_grad_fns(
                    kwargs.values()
                )
                context = EventContext(tools=[_tool])
                context.trigger(
                    before_op_executed,
                    op=func,
                    args=args,
                    kwargs=kwargs,
                )
                output = func(*args, **kwargs)
                context.trigger(
                    after_op_executed,
                    op=func,
                    output=output,
                )
                context.register_bw_events_recursively(output, input_grad_fns)
                return context["output"]
        else:
            return func(*args, **kwargs)

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


TORCH_OP_LIST = set()


def listener_callback(op_name: str) -> str:
    def remove_namespace(name: str) -> str:
        pos = name.find("::")
        if not pos == -1:
            return name[pos + 2 :]
        else:
            return name

    global TORCH_OP_LIST
    TORCH_OP_LIST.add(remove_namespace(op_name))
    return op_name


class ListenerFunctionalUpdater(MatchedFunctionUpdater):
    def __init__(self, modules: List[str], submodules: List[str]):
        super().__init__(module="", decorator=function_wrapper)
        self.modules = modules
        self.submodules = submodules

    def is_match(self, name: str) -> bool:
        return name in self.modules

    def is_match_func(self, module: str, name: str, func) -> bool:
        pass

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
                        if func_key.startswith("__") or func_key not in TORCH_OP_LIST:
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
                        if func_key.startswith("__") or func_key not in TORCH_OP_LIST:
                            continue
                        # print(module.__name__, submodule_cls_key, func_key)
                        self.update_object(
                            module, submodule_key, submodule_cls_key, func_key
                        )
                else:
                    funcs = dict(module.__dict__[submodule_key].__dict__)
                    for func_key in funcs:
                        if func_key.startswith("__") or func_key not in TORCH_OP_LIST:
                            continue
                        # print(module.__name__, submodule_key, func_key)
                        self.update_module(module, submodule_key, func_key)


class FakeUpdater(MatchedFunctionUpdater):
    def __init__(self):
        super().__init__(module="", decorator=function_wrapper)

    def is_match(self, name: str) -> bool:
        return True

    def is_match_func(self, module: str, name: str, func) -> bool:
        return name == "conv2d"

    def update(self, module) -> None:
        funcs = dict(module.__dict__)
        for name in funcs:
            func = funcs[name]
            if self.is_match_func(module.__name__, name, func):
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                print(module.__name__, name)


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
    # register_updater(FakeUpdater())
    # register_updater(
    #     FunctionalUpdater(
    #         modules=[
    #             "torch.nn.functional",
    #             "torch.nn.quantized.functional",
    #             # "torch.nn.init",
    #             "torch._C._nn",
    #         ]
    #     )
    # )
    register_updater(
        ListenerFunctionalUpdater(
            modules=[
                "torch._C",
            ],
            submodules=[
                "_nn",
                "_fft",
                "_linalg",
                "_TensorBase",
                "_VariableFunctions",
            ],
        )
    )
    # register_updater(ModuleUpdater())
    # register_updater(GradFnUpdater())


def register_listener():
    from amanda.conversion.listener.build.listener import HookRegisterer

    HookRegisterer(listener_callback)


_tool: Tool = None


@contextmanager
def apply(tool: Tool):
    global _tool
    _tool = tool
    yield
    _tool = None
