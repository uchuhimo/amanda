import atexit
import ctypes
import sys
import types
from functools import partial
from typing import Callable, Dict, List

from amanda.amanda_intercepts_pybind import (
    addr,
    get_builtin_handler,
    get_getset_descriptor_handler,
    get_method_descriptor_handler,
)
from intercepts.functypes import PyCFunctionObject, PyMethodDef, PyObject, PyTypeObject
from intercepts.utils import (
    copy_builtin,
    copy_function,
    create_code_like,
    replace_builtin,
    replace_function,
    update_wrapper,
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


_HANDLERS: Dict[int, List[Callable]] = {}


def _func_handler(*args, **kwargs):
    consts = sys._getframe(0).f_code.co_consts
    func_id = consts[-1]
    _func = _HANDLERS[func_id][0]
    handler = _func
    # assert len(_HANDLERS[func_id]) == 3
    for _handler in _HANDLERS[func_id][2:]:
        handler = update_wrapper(partial(_handler, handler), _func)
    result = handler(*args, **kwargs)
    return result


def _getter_handler(*args, **kwargs):
    consts = sys._getframe(0).f_code.co_consts
    func_id = consts[-1]
    _func = _HANDLERS[func_id][0]
    handler = _func.__get__
    # assert len(_HANDLERS[func_id]) == 3
    for _handler in _HANDLERS[func_id][2:]:
        handler = update_wrapper(partial(_handler, handler), _func)
    result = handler(*args, **kwargs)
    return result


def _setter_handler(*args, **kwargs):
    consts = sys._getframe(0).f_code.co_consts
    func_id = consts[-1]
    _func = _HANDLERS[func_id][0]
    handler = _func.__set__
    # assert len(_HANDLERS[func_id]) == 3
    for _handler in _HANDLERS[func_id][2:]:
        handler = update_wrapper(partial(_handler, handler), _func)
    result = handler(*args, **kwargs)
    return result


def create_handler_with_addr(handler, func_addr):
    handler_code = create_code_like(
        handler.__code__,
        consts=(handler.__code__.co_consts + (func_addr,)),
        name=handler.__name__,
    )
    global_dict = handler.__globals__  # type: ignore
    new_handler = types.FunctionType(
        handler_code,
        global_dict,
        handler.__name__,
        handler.__defaults__,
        handler.__closure__,
    )
    new_handler.__code__ = handler_code
    return new_handler


def register_builtin(func, handler):
    func_addr = addr(func)
    if func_addr not in _HANDLERS:
        func_copy = PyCFunctionObject()
        copy_builtin(addr(func_copy), func_addr)
        handler_with_addr = create_handler_with_addr(_func_handler, func_addr)
        _handler = get_builtin_handler(func_addr, handler_with_addr)
        _HANDLERS[func_addr] = [func_copy, _handler]
        replace_builtin(func_addr, addr(_handler))
    _HANDLERS[func_addr].append(handler)
    return func


def register_function(
    func: types.FunctionType, handler: types.FunctionType
) -> types.FunctionType:
    r"""Registers an intercept handler for a function.

    :param func: The function to intercept.
    :param handler: A function to handle the intercept.
    """
    func_addr = addr(func)
    if func_addr not in _HANDLERS:
        handler_code = create_code_like(
            _func_handler.__code__,
            consts=(_func_handler.__code__.co_consts + (func_addr,)),
            name=func.__name__,
        )
        global_dict = _func_handler.__globals__  # type: ignore
        _handler = types.FunctionType(
            handler_code,
            global_dict,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        _handler.__code__ = handler_code

        handler_addr = addr(_handler)

        def func_copy(*args, **kwargs):
            pass

        copy_function(addr(func_copy), func_addr)

        _HANDLERS[func_addr] = [func_copy, _handler]
        replace_function(func_addr, handler_addr)
    _HANDLERS[func_addr].append(handler)
    return func


def register_method(
    method: types.MethodType, handler: types.FunctionType
) -> types.MethodType:
    r"""Registers an intercept handler for a method.

    :param method: The method to intercept.
    :param handler: A function to handle the intercept.
    """
    register_function(method.__func__, handler)
    return method


def register_method_descriptor(func, handler):
    func_addr = addr(func)
    if func_addr not in _HANDLERS:
        func_copy = PyMethodDescrObject()
        copy_method_descriptor(addr(func_copy), func_addr)
        handler_with_addr = create_handler_with_addr(_func_handler, func_addr)
        _handler = get_method_descriptor_handler(func_addr, handler_with_addr)
        _HANDLERS[func_addr] = [func_copy, _handler]
        replace_method_descriptor(func_addr, addr(_handler))
    _HANDLERS[func_addr].append(handler)
    return func


def register_getset_descriptor(attribute, handler):
    func_addr = addr(attribute)
    if func_addr not in _HANDLERS:
        func_copy = PyGetSetDescrObject()
        copy_getset_descriptor(addr(func_copy), func_addr)
        getter_handler_with_addr = create_handler_with_addr(_getter_handler, func_addr)
        setter_handler_with_addr = create_handler_with_addr(_setter_handler, func_addr)
        _handler = get_getset_descriptor_handler(
            func_addr,
            getter_handler_with_addr,
            setter_handler_with_addr,
        )
        _HANDLERS[func_addr] = [func_copy, _handler]
        replace_method_descriptor(func_addr, addr(_handler))
    _HANDLERS[func_addr].append(handler)
    return attribute


def to_handler(wrapper):
    def handler(func, *args, **kwargs):
        return wrapper(func)(*args, **kwargs)

    return handler


def register(obj, handler):
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

    if isinstance(obj, types.BuiltinFunctionType):
        return register_builtin(obj, handler)
    elif isinstance(obj, types.FunctionType):
        return register_function(obj, handler)
    elif isinstance(obj, types.MethodType):
        return register_method(obj, handler)
    elif isinstance(obj, types.MethodDescriptorType):
        return register_method_descriptor(obj, handler)
    elif isinstance(obj, types.GetSetDescriptorType):
        return register_getset_descriptor(obj, handler)
    else:
        raise NotImplementedError(f"{obj} has unsupported type: {type(obj)}")


def unregister(obj, depth: int = -1):
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
    if depth < 0 or len(handlers) - depth <= 2:
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
    else:
        _HANDLERS[func_addr] = handlers[:-depth]
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
    import importlib.util

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


init()
