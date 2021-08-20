import importlib.machinery
import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from importlib._bootstrap_external import FileLoader
from typing import Callable, List, Type

import _imp
from loguru import logger

from amanda.lang import Handler, register_handler
from amanda.threading import ThreadLocalStack


class Updater(ABC):
    @abstractmethod
    def is_match(self, name: str) -> bool:
        pass

    @abstractmethod
    def update(self, module) -> None:
        pass


def update_method(updater, cls):
    method = getattr(cls, updater.method)
    if hasattr(method, "updated") and method.updated:
        return
    new_method = updater.decorator(method)
    if hasattr(new_method, "__dict__"):
        new_method.original = method
        new_method.updated = True
    logger.debug("hook method: {}.{}.{}", cls.__module__, cls.__name__, method.__name__)
    setattr(cls, updater.method, new_method)


@dataclass
class MethodUpdater(Updater):
    module: str
    cls: str
    method: str
    decorator: Callable

    def is_match(self, name: str) -> bool:
        return name == self.module

    def update(self, module) -> None:
        cls: Type = module.__dict__[self.cls]
        update_method(self, cls)


class MatchedClassUpdater(Updater):
    def __init__(self, module: str, method: str, decorator: Callable):
        self.module: str = module
        self.method: str = method
        self.decorator: Callable = decorator

    def is_match(self, name: str) -> bool:
        return self.module.startswith(name)

    @abstractmethod
    def is_match_class(self, module: str, name: str, cls) -> bool:
        pass

    def update(self, module) -> None:
        classes = dict(module.__dict__)
        for name in classes:
            cls: Type = classes[name]
            if inspect.isclass(cls) and self.is_match_class(module.__name__, name, cls):
                update_method(self, cls)


def update_func(updater, module, func_name):
    func = module.__dict__[func_name]
    if hasattr(func, "updated") and func.updated:
        return
    new_func = updater.decorator(func)
    new_func.original = func
    new_func.updated = True
    logger.debug("hook function: {}.{}", module.__name__, func.__name__)
    module.__dict__[func_name] = new_func


@dataclass
class FunctionUpdater(Updater):
    module: str
    func: str
    decorator: Callable

    def is_match(self, name: str) -> bool:
        return name == self.module

    def update(self, module) -> None:
        update_func(self, module, self.func)


class MatchedFunctionUpdater(Updater):
    def __init__(self, module: str, decorator: Callable):
        self.module: str = module
        self.decorator: Callable = decorator

    def is_match(self, name: str) -> bool:
        return self.module.startswith(name)

    @abstractmethod
    def is_match_func(self, module: str, name: str, func) -> bool:
        pass

    def update(self, module) -> None:
        funcs = dict(module.__dict__)
        for name in funcs:
            func = funcs[name]
            if inspect.isfunction(func) and self.is_match_func(
                module.__name__, name, func
            ):
                update_func(self, module, name)


_updaters: List[Updater] = []


def register_updater(updater: Updater) -> None:
    _updaters.append(updater)


class HookedLoader(FileLoader):
    def __init__(self, fullname, path):
        super().__init__(fullname, path)

    def exec_module(self, module):
        super().exec_module(module)
        for updater in _updaters:
            if updater.is_match(self.name):
                updater.update(module)


def log_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"hook {func}")
        return func(*args, **kwargs)

    return wrapper


class SourceFileLoader(HookedLoader, importlib.machinery.SourceFileLoader):
    pass


class SourcelessFileLoader(HookedLoader, importlib.machinery.SourcelessFileLoader):
    pass


class ExtensionFileLoader(HookedLoader, importlib.machinery.ExtensionFileLoader):
    pass


def init_import_hook() -> None:
    extensions = ExtensionFileLoader, _imp.extension_suffixes()
    source = SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES
    bytecode = SourcelessFileLoader, importlib.machinery.BYTECODE_SUFFIXES
    path_hook = importlib.machinery.FileFinder.path_hook(
        extensions, source, bytecode  # type: ignore
    )
    sys.path_hooks.insert(0, path_hook)


class InstScopeHook:
    def begin(self, is_enabled: bool) -> None:
        ...

    def end(self, is_enabled: bool) -> None:
        ...


_inst_scope_hooks: List[InstScopeHook] = []


def register_inst_scope_hook(hook: InstScopeHook) -> Handler:
    return register_handler(_inst_scope_hooks, hook)


_enabled = ThreadLocalStack()


@contextmanager
def disabled():
    _enabled.push(False)
    for hook in _inst_scope_hooks:
        hook.begin(False)
    try:
        yield
    finally:
        for hook in _inst_scope_hooks:
            hook.end(False)
        _enabled.pop()


@contextmanager
def enabled():
    _enabled.push(True)
    for hook in _inst_scope_hooks:
        hook.begin(True)
    try:
        yield
    finally:
        for hook in _inst_scope_hooks:
            hook.end(True)
        _enabled.pop()


def is_enabled() -> bool:
    return _enabled.top() or _enabled.top() is None


_inited: bool = False


def check_enabled(func, hooked_func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from amanda.tool import get_tools

        if is_enabled():
            tools = get_tools()
            if len(tools) == 0:
                return func(*args, **kwargs)
            else:
                return hooked_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def init() -> None:
    global _inited
    if _inited:
        return
    if importlib.util.find_spec("torch"):
        from .conversion.pytorch_updater import (
            register_import_hook as pytorch_register_import_hook,
        )
        from .conversion.pytorch_updater import (
            register_listener as pytorch_register_listener,
        )

        pytorch_register_import_hook()
        pytorch_register_listener()
    if importlib.util.find_spec("tensorflow"):
        from .conversion.tensorflow_updater import (
            register_import_hook as tensorflow_register_import_hook,
        )

        tensorflow_register_import_hook()
    init_import_hook()
    from amanda.conversion.listener.build.amanda_pybind import init_THPVariableClass

    init_THPVariableClass()
    _inited = True


init()
