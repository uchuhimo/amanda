import ctypes
import gc
import inspect
import sys
from ctypes import pythonapi as api
from types import (
    BuiltinFunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodType,
)
from typing import Any, List, Type

import guppy
from guppy.heapy import Path
from loguru import logger

hp = guppy.hpy()


def _w(x):
    def f():
        x

    return f


CellType = type(_w(0).__closure__[0])

del _w


def _write_struct_attr(addr, value, add_offset):
    ptr_size = ctypes.sizeof(ctypes.py_object)
    ptrs_in_struct = (3 if hasattr(sys, "getobjects") else 1) + add_offset
    offset = ptrs_in_struct * ptr_size + ctypes.sizeof(ctypes.c_ssize_t)
    ref = ctypes.byref(ctypes.py_object(value))
    ctypes.memmove(addr + offset, ref, ptr_size)


def _replace_attribute(source, rel, old, new):
    if isinstance(source, (MethodType, BuiltinFunctionType)):
        if rel == "__self__":
            # Note: PyMethodObject->im_self and PyCFunctionObject->m_self
            # have the same offset
            _write_struct_attr(id(source), new, 1)
            return
        if rel == "im_self":
            return  # Updated via __self__
    if isinstance(source, type):
        if rel == "__base__":
            return  # Updated via __bases__
        if rel == "__mro__":
            return  # Updated via __bases__ when important, otherwise futile
    if isinstance(source, (GetSetDescriptorType, MemberDescriptorType)):
        if rel == "__objclass__":
            _write_struct_attr(id(source), new, 0)
            return
    try:
        setattr(source, rel, new)
    except TypeError:
        logger.warning(f"Unknown R_ATTRIBUTE (read-only): {rel}, {type(source)}")


def isinstance_namedtuple(obj) -> bool:
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def _replace_indexval(source, rel, old, new):
    if isinstance(source, tuple):
        temp = list(source)
        temp[rel] = new
        if isinstance_namedtuple(source):
            replace_all_refs(source, type(source)._make(temp))
        else:
            replace_all_refs(source, tuple(temp))
        return
    source[rel] = new


def _replace_indexkey(source, rel, old, new):
    source[new] = source.pop(old)


def _replace_interattr(source, rel, old, new):
    if isinstance(source, CellType):
        api.PyCell_Set(ctypes.py_object(source), ctypes.py_object(new))
        return
    if rel == "ob_type":
        source.__class__ = new
        return
    logger.warning(f"Unknown R_INTERATTR: {rel}, {type(source)}")


def _replace_local_var(source, rel, old, new):
    source.f_locals[rel] = new
    api.PyFrame_LocalsToFast(ctypes.py_object(source), ctypes.c_int(0))


def _replace_inset(source, rel, old, new):
    assert old in source
    source.remove(old)
    assert new not in source
    source.add(new)


_RELATIONS = {
    Path.R_ATTRIBUTE: _replace_attribute,
    Path.R_INDEXVAL: _replace_indexval,
    Path.R_INDEXKEY: _replace_indexkey,
    Path.R_INTERATTR: _replace_interattr,
    Path.R_CELL: _replace_local_var,
    Path.R_LOCAL_VAR: _replace_local_var,
    Path.R_INSET: _replace_inset,
}


def get_all_refs(target):
    gc_referrers = hp.idset(gc.get_referrers(target))
    idset = hp.iso(target)
    paths = list(idset.get_shpaths(gc_referrers | idset.referrers))
    return paths


def replace_all_refs(old, new, exclude_caller: bool = False):
    gc_referrers = hp.idset(gc.get_referrers(old))
    idset = hp.iso(old)
    paths = list(idset.get_shpaths(gc_referrers | idset.referrers))
    if exclude_caller:
        current_frame = inspect.currentframe()
        caller_frame = inspect.stack()[1].frame
        f_locals = current_frame.f_locals
        caller_f_locals = caller_frame.f_locals
        excluded_source = [current_frame, caller_frame, f_locals, caller_f_locals]
    else:
        excluded_source = []
    for path in paths:
        source = path.src.theone
        if source in excluded_source:
            continue
        relation = path.path[1]
        try:
            func = _RELATIONS[type(relation).__bases__[0]]
        except KeyError:
            logger.warning(
                f"Unknown relation: "
                f"{type(relation).__bases__[0]}, {relation}, {type(source)}, {source}"
            )
            continue
        func(source, relation.r, old, new)


def get_superclasses(cls: Type) -> List[str]:
    return [
        f"{superclass.__module__}.{superclass.__name__}" for superclass in cls.__mro__
    ]


class Handler:
    def __init__(self, container: List[Any], element: Any) -> None:
        self.container = container
        self.element = element
        container.append(element)

    def unregister(self):
        self.container.remove(self.element)


def register_handler(container: List[Any], element: Any) -> Handler:
    return Handler(container, element)
