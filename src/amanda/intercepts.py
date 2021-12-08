import atexit
import ctypes
import importlib.util
import sys
import types
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Tuple

from intercepts.functypes import PyCFunctionObject, PyMethodDef, PyObject, PyTypeObject
from intercepts.utils import (
    copy_builtin,
    copy_function,
    create_code_like,
    replace_builtin,
    replace_function,
    update_wrapper,
)

from amanda.amanda_intercepts_pybind import (
    addr,
    get_builtin_handler,
    get_getset_descriptor_handler,
    get_method_descriptor_handler,
)


class wrapperbase(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("offset", ctypes.c_int),
        ("function", ctypes.c_void_p),
        ("wrapper", ctypes.c_void_p),
        ("doc", ctypes.c_char_p),
        ("flags", ctypes.c_int),
        ("name_strobj", ctypes.POINTER(PyObject)),
    ]


class PyMethodDescrObject(ctypes.Structure):
    _fields_ = [
        ("ob_base", PyObject),
        ("d_type", ctypes.POINTER(PyTypeObject)),
        ("d_name", ctypes.POINTER(PyObject)),
        ("d_qualname", ctypes.POINTER(PyObject)),
        ("d_method", ctypes.POINTER(PyMethodDef)),
    ]


class PyGetSetDef(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("get", ctypes.c_void_p),
        ("set", ctypes.c_void_p),
        ("doc", ctypes.c_char_p),
        ("closure", ctypes.c_void_p),
    ]


class PyGetSetDescrObject(ctypes.Structure):
    _fields_ = [
        ("ob_base", PyObject),
        ("d_type", ctypes.POINTER(PyTypeObject)),
        ("d_name", ctypes.POINTER(PyObject)),
        ("d_qualname", ctypes.POINTER(PyObject)),
        ("d_getset", ctypes.POINTER(PyGetSetDef)),
    ]


class PyWrapperDescrObject(ctypes.Structure):
    _fields_ = [
        ("ob_base", PyObject),
        ("d_type", ctypes.POINTER(PyTypeObject)),
        ("d_name", ctypes.POINTER(PyObject)),
        ("d_qualname", ctypes.POINTER(PyObject)),
        ("d_base", ctypes.POINTER(wrapperbase)),
        ("d_wrapped", ctypes.c_void_p),
    ]


def copy_method_descriptor(dst, src):
    ctypes.memmove(dst, src, 48)


def replace_method_descriptor(dst, src):
    ctypes.memmove(dst + 16, src + 16, 32)


def copy_getset_descriptor(dst, src):
    ctypes.memmove(dst, src, 48)


def replace_getset_descriptor(dst, src):
    ctypes.memmove(dst + 16, src + 16, 32)


_HANDLERS: Dict[int, Tuple[Callable, Callable, Dict[Any, Callable]]] = {}


def func_handler(func_id, *args, **kwargs):
    func = _HANDLERS[func_id][0]
    updated_func = func
    for _handler in _HANDLERS[func_id][2].values():
        updated_func = update_wrapper(partial(_handler, updated_func), func)
    result = updated_func(*args, **kwargs)
    return result


def func_handler_prototype(*args, **kwargs):
    consts = sys._getframe(0).f_code.co_consts
    func_id = consts[-1]
    func = _HANDLERS[func_id][0]
    updated_func = func
    for _handler in _HANDLERS[func_id][2].values():
        updated_func = update_wrapper(partial(_handler, updated_func), func)
    result = updated_func(*args, **kwargs)
    return result


def getter_handler(func_id, *args, **kwargs):
    func = _HANDLERS[func_id][0].__get__
    updated_func = func
    for _handler in _HANDLERS[func_id][2].values():
        updated_func = update_wrapper(partial(_handler, updated_func), func)
    result = updated_func(*args, **kwargs)
    return result


def setter_handler(func_id, *args, **kwargs):
    func = _HANDLERS[func_id][0].__set__
    updated_func = func
    for _handler in _HANDLERS[func_id][2].values():
        updated_func = update_wrapper(partial(_handler, updated_func), func)
    result = updated_func(*args, **kwargs)
    return result


def register_builtin(func, handler, key):
    func_addr = addr(func)
    if func_addr not in _HANDLERS:
        func_copy = PyCFunctionObject()
        copy_builtin(addr(func_copy), func_addr)
        handler_with_addr = partial(func_handler, func_addr)
        new_func = get_builtin_handler(func, handler_with_addr)
        _HANDLERS[func_addr] = (func_copy, new_func, OrderedDict())
        replace_builtin(func_addr, addr(new_func))
    _HANDLERS[func_addr][2][key] = handler
    if len(_HANDLERS[func_addr][2]) > 1:
        print(_HANDLERS[func_addr][2])
    return func


def register_function(
    func: types.FunctionType, handler: types.FunctionType, key
) -> types.FunctionType:
    func_addr = addr(func)
    if func_addr not in _HANDLERS:

        def func_copy(*args, **kwargs):
            pass

        copy_function(addr(func_copy), func_addr)
        handler_code = create_code_like(
            func_handler_prototype.__code__,
            consts=(func_handler_prototype.__code__.co_consts + (func_addr,)),
            name=func.__name__,
        )
        global_dict = func_handler_prototype.__globals__  # type: ignore
        new_func = types.FunctionType(
            handler_code,
            global_dict,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        new_func.__code__ = handler_code
        _HANDLERS[func_addr] = (func_copy, new_func, OrderedDict())
        replace_function(func_addr, addr(new_func))
    _HANDLERS[func_addr][2][key] = handler
    if len(_HANDLERS[func_addr][2]) > 1:
        print(_HANDLERS[func_addr][2])
    return func


def register_method(
    method: types.MethodType, handler: types.FunctionType, key
) -> types.MethodType:
    register_function(method.__func__, handler, key)
    return method


def register_method_descriptor(func, handler, key):
    func_addr = addr(func)
    if func_addr not in _HANDLERS:
        func_copy = PyMethodDescrObject()
        copy_method_descriptor(addr(func_copy), func_addr)
        handler_with_addr = partial(func_handler, func_addr)
        new_func = get_method_descriptor_handler(func, handler_with_addr)
        _HANDLERS[func_addr] = (func_copy, new_func, OrderedDict())
        replace_method_descriptor(func_addr, addr(new_func))
    _HANDLERS[func_addr][2][key] = handler
    if len(_HANDLERS[func_addr][2]) > 1:
        print(_HANDLERS[func_addr][2])
    return func


def register_getset_descriptor(attribute, handler, key):
    func_addr = addr(attribute)
    if func_addr not in _HANDLERS:
        func_copy = PyGetSetDescrObject()
        copy_getset_descriptor(addr(func_copy), func_addr)
        getter_handler_with_addr = partial(getter_handler, func_addr)
        setter_handler_with_addr = partial(setter_handler, func_addr)
        new_func = get_getset_descriptor_handler(
            attribute,
            getter_handler_with_addr,
            setter_handler_with_addr,
        )
        _HANDLERS[func_addr] = (func_copy, new_func, OrderedDict())
        replace_method_descriptor(func_addr, addr(new_func))
    _HANDLERS[func_addr][2][key] = handler
    if len(_HANDLERS[func_addr][2]) > 1:
        print(_HANDLERS[func_addr][2])
    return attribute


def to_handler(wrapper):
    def handler(func, *args, **kwargs):
        return wrapper(func)(*args, **kwargs)

    return handler


def register(obj, handler, key=None):
    r"""Registers an intercept handler.

    :param obj: The callable to intercept.
    :param handler: A function to handle the intercept.

    Usage::

        >>> from amanda import intercepts
        >>> increment = lambda x: x + 1
        >>> handler = lambda func, arg: arg - (func(arg) - arg)
        >>> intercepts.register(increment, handler)
        >>> increment(43)
        42
    """
    if not isinstance(handler, types.FunctionType):
        raise ValueError("Argument `handler` must be a function.")
    if obj == handler:
        raise ValueError("A function cannot handle itself")

    key = key or id(handler)
    if isinstance(obj, types.BuiltinFunctionType):
        return register_builtin(obj, handler, key)
    elif isinstance(obj, types.FunctionType):
        return register_function(obj, handler, key)
    elif isinstance(obj, types.MethodType):
        return register_method(obj, handler, key)
    elif isinstance(obj, types.MethodDescriptorType):
        return register_method_descriptor(obj, handler, key)
    elif isinstance(obj, types.GetSetDescriptorType):
        return register_getset_descriptor(obj, handler, key)
    else:
        raise NotImplementedError(f"{obj} has unsupported type: {type(obj)}")


def unregister(obj):
    r"""Unregisters the handlers for an object.

    :param obj: The callable for which to unregister handlers.
    :param depth: (optional) The maximum number of handlers to unregister.
    """
    # TODO : use an isinstance replacement
    if isinstance(obj, (types.BuiltinFunctionType, types.FunctionType)):
        func_addr = addr(obj)
    else:
        func_addr = addr(obj.__func__)
    handlers = _HANDLERS[func_addr]
    orig_func = handlers[0]
    if isinstance(orig_func, types.BuiltinFunctionType):
        replace_builtin(func_addr, addr(orig_func))
    elif isinstance(orig_func, types.FunctionType):
        replace_function(func_addr, addr(orig_func))
    elif isinstance(orig_func, types.MethodDescriptorType):
        replace_method_descriptor(func_addr, addr(orig_func))
    elif isinstance(orig_func, types.GetSetDescriptorType):
        replace_getset_descriptor(func_addr, addr(orig_func))
    else:
        raise ValueError("Unknown type of handled function: %s" % type(orig_func))
    del _HANDLERS[func_addr]
    assert func_addr not in _HANDLERS
    return obj


@atexit.register
def unregister_all() -> None:
    r"""Unregisters all handlers."""
    global _HANDLERS
    for func_addr, handlers in _HANDLERS.items():
        orig_func = handlers[0]
        if isinstance(orig_func, types.BuiltinFunctionType):
            replace_builtin(func_addr, addr(orig_func))
        elif isinstance(orig_func, types.FunctionType):
            replace_function(func_addr, addr(orig_func))
        elif isinstance(orig_func, types.MethodDescriptorType):
            replace_method_descriptor(func_addr, addr(orig_func))
        elif isinstance(orig_func, types.GetSetDescriptorType):
            replace_getset_descriptor(func_addr, addr(orig_func))
        else:
            raise ValueError("Unknown type of handled function: %s" % type(orig_func))
    _HANDLERS = {}


_inited: bool = False


def init() -> None:
    global _inited
    if _inited:
        return
    if importlib.util.find_spec("torch"):
        from .conversion.pytorch_updater import (
            register_intercepts as pytorch_register_intercepts,
        )

        pytorch_register_intercepts()
    if importlib.util.find_spec("tensorflow"):
        from .conversion.tensorflow_updater import (
            register_intercepts as tensorflow_register_intercepts,
        )

        tensorflow_register_intercepts()
    _inited = True
