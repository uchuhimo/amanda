from typing import List


class Tensor:
    def __init__(
        self, input: "Op", dtype=None, shape=None, value=None, attrs={},
    ):
        self.input = input
        self.attrs = attrs
        self.dtype = dtype
        self.shape = shape
        self.value = value


class Op:
    def __init__(self, name, type, inputs: List[Tensor], attrs={}):
        self.name = name
        self.type = type
        self.inputs = inputs
        self.attrs = attrs


class Graph:
    def __init__(self, ops: List[Op]):
        self.ops = ops
