from typing import Any, List

import torch
from torch import Tensor


class Op:
    raw_op: Any
    input_ops: List[Any]
    inputs: List[Tensor]
    outputs: List[Tensor]

    def __init__(self, raw_op, input_ops, inputs, outputs):
        self.raw_op = raw_op
        self.input_ops = input_ops
        self.inputs = inputs
        self.outputs = outputs


class Graph:
    ops: List[Op]

    def __init__(self):
        self.ops = list()
