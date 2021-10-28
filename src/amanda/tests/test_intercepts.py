import torch

from amanda import intercepts


def func(x):
    return x


def test_register_function():
    def handler(func, *args, **kwargs):
        nonlocal count
        count += 1
        return func(*args, **kwargs)

    count = 0
    for _ in range(10):
        intercepts.register(func, handler)
    for _ in range(10):
        assert func(1) == 1
    assert count == 100


class A:
    def func(self, x):
        return x


def test_register_method():
    def handler(func, *args, **kwargs):
        nonlocal count
        count += 1
        return func(*args, **kwargs)

    count = 0
    intercepts.register(A.func, handler)
    for _ in range(10):
        assert A().func(1) == 1
    assert count == 10


def test_register_builtin():
    def handler(func, *args, **kwargs):
        nonlocal count
        count += 1
        return func(*args, **kwargs)

    count = 0
    intercepts.register(torch.abs, handler)
    x = torch.randn((3, 3))
    for _ in range(10):
        torch.abs(x)
    assert count == 10


def test_register_method_descriptor():
    def handler(func, *args, **kwargs):
        nonlocal count
        count += 1
        return func(*args, **kwargs)

    count = 0
    intercepts.register(torch.Tensor.abs, handler)
    x = torch.randn((3, 3))
    for _ in range(10):
        x.abs()
    assert count == 10


def test_register_both():
    def handler(func, *args, **kwargs):
        nonlocal count
        count += 1
        return func(*args, **kwargs)

    count = 0
    intercepts.register(torch.abs, handler)
    intercepts.register(torch.Tensor.abs, handler)
    x = torch.randn((3, 3))
    for _ in range(10):
        torch.abs(x)
        x.abs()
    assert count == 20


def test_register_getset_descriptor_both():
    def handler(func, *args, **kwargs):
        nonlocal count
        count += 1
        return func(*args, **kwargs)

    count = 0
    intercepts.register(torch.Tensor.data, handler)
    x = torch.randn((3, 3))
    for _ in range(10):
        x.data = x.data
    assert count == 20


def test_register_getset_descriptor_getter():
    def handler(func, *args, **kwargs):
        nonlocal count
        count += 1
        return func(*args, **kwargs)

    count = 0
    intercepts.register(torch.Tensor.is_sparse, handler)
    x = torch.randn((3, 3))
    for _ in range(10):
        x.is_sparse
    assert count == 10
