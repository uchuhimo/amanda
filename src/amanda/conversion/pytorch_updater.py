from contextlib import contextmanager
from functools import wraps
from typing import List

from amanda.event import EventContext, after_op_executed, before_op_executed
from amanda.import_hook import (
    MatchedClassUpdater,
    MatchedFunctionUpdater,
    MethodUpdater,
    register_updater,
)
from amanda.lang import get_superclasses
from amanda.tool import Tool


def function_wrapper(func, pass_type=None):
    @wraps(func)
    def wrapper(*args, **kwargs):
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
        return context["output"]

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
    register_updater(
        FunctionalUpdater(
            modules=[
                "torch.nn.functional",
                "torch.nn.quantized.functional",
                # "torch.nn.init",
                "torch._C._nn",
            ]
        )
    )
    # register_updater(ModuleUpdater())
    # register_updater(GradFnUpdater())


_tool: Tool = None


@contextmanager
def apply(tool: Tool):
    global _tool
    _tool = tool
    yield
    _tool = None
