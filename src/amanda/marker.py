import functools
import threading
import types
import weakref
from contextlib import contextmanager

from amanda.graph import Graph


class ThreadLocalStack(threading.local):
    """A thread-local stack."""

    def __init__(self):
        super(ThreadLocalStack, self).__init__()
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        self.stack.pop()

    def top(self):
        return self.stack[-1]

    def __iter__(self):
        return reversed(self.stack).__iter__()

    def swap_stack(self, new_stack):
        old_stack = self.stack
        self.stack = new_stack
        return old_stack


_graph_stack = ThreadLocalStack()


def instrumentation(func):
    def wrapper(*args, **kw):
        if not args:
            raise TypeError(f"{funcname} requires at least 1 positional argument")
        if not isinstance(args[0], Graph):
            raise TypeError(
                f"the first argument's type should be amanda.Graph, "
                f"get {args[0]} instead"
            )

        @contextmanager
        def default_graph():
            _graph_stack.push(args[0])
            try:
                yield
            finally:
                _graph_stack.pop()

        with default_graph():
            return func(*args, **kw)

    funcname = getattr(func, "__name__", "instrumentation function")
    functools.update_wrapper(wrapper, func)
    return wrapper


def dispatch(func):

    registry = {}
    default_impl = None
    dispatch_cache = weakref.WeakKeyDictionary()

    def dispatch(namespace):
        if namespace in registry:
            return registry[namespace]
        if namespace in dispatch_cache:
            return dispatch_cache[namespace]
        for registered_namespace in registry:
            if namespace.belong_to(registered_namespace):
                dispatch_cache[namespace] = registry[registered_namespace]
                return dispatch_cache[namespace]
        return default_impl

    def register(namespace, func=None):
        def wrapper(func):
            registry[namespace] = func
            dispatch_cache.clear()
            return func

        if func is None:
            return wrapper
        else:
            return wrapper(func)

    def wrapper(*args, **kw):
        return dispatch(_graph_stack.top().namespace)(*args, **kw)

    default_impl = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = types.MappingProxyType(registry)
    wrapper._clear_cache = dispatch_cache.clear
    functools.update_wrapper(wrapper, func)
    return wrapper
