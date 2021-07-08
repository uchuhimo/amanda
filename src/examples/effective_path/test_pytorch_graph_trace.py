import amanda
import pytest
import torch
import torchvision
from trace_tool import TraceEffectivePathTool

from amanda.conversion.pytorch_updater import apply

""" graph_traverse(utils.Graph) -> None
traverse a fw graph traced by TraceEffectivePathTool 
    in backward fashion, but not backward propagation,
forward graph with residual graph may cause deep loop,
    so each op is visited for once,
"""


def graph_traverse(graph):
    def _dfs(op):
        print(op.raw_op.__name__)
        visited.append(op)
        for input_op in op.input_ops:
            if input_op in visited:
                continue
            _dfs(input_op)

    op = graph.ops[-1]

    visited = list()

    _dfs(op)


def test_graph_trace():
    TEST_MODELS = {
        "resnet": torchvision.models.resnet50,
        "inception": torchvision.models.inception_v3,
        "vgg": torchvision.models.vgg19_bn,
    }

    model = TEST_MODELS["resnet"]()

    x = torch.rand((4, 3, 500, 500))

    tracer = TraceEffectivePathTool()

    with apply(tracer):
        model(x)

    graph_traverse(tracer.graph)
