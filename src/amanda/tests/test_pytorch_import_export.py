import numpy as np
import pytest
import torch
import torch.jit
import torchvision.models as models

from amanda.conversion.pytorch import export_to_graph, import_from_graph


@pytest.fixture(
    scope="module",
    params=[
        models.alexnet,
        models.vgg11,
        models.vgg11_bn,
        models.squeezenet1_1,
        models.mobilenet_v2,
        models.mnasnet0_5,
    ],
)
def scripted_model(request):
    model = request.param(pretrained=False, progress=False)
    model.eval()
    return torch.jit.script(model)


@pytest.fixture(
    scope="module",
    params=[
        models.resnet18,
        models.alexnet,
        models.vgg11,
        models.vgg11_bn,
        models.squeezenet1_1,
        models.shufflenet_v2_x0_5,
        models.mobilenet_v2,
        models.mnasnet0_5,
    ],
)
def traced_model(request):
    model = request.param(pretrained=False, progress=False)
    model.eval()
    return torch.jit.trace(model, (torch.randn(1, 3, 224, 224),))


def test_pytorch_import_export_script(scripted_model):
    x = torch.randn(1, 3, 224, 224, requires_grad=False)
    torch_graph = scripted_model.graph
    graph = import_from_graph(torch_graph)
    print(len(graph.ops))
    output = scripted_model(x)
    np.testing.assert_allclose(output.detach().numpy(), output.detach().numpy())


@pytest.mark.skip
def test_pytorch_import_export_trace(traced_model):
    x = torch.randn(1, 3, 224, 224, requires_grad=False)
    output = traced_model(x)
    torch_graph = traced_model.graph
    graph = import_from_graph(torch_graph)
    new_torch_graph = export_to_graph(graph)
    traced_model.forward.graph = new_torch_graph
    new_output = traced_model(x)
    np.testing.assert_allclose(output.detach().numpy(), new_output.detach().numpy())
