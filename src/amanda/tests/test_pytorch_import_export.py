import numpy as np
import pytest
import torch
import torch.jit
import torchvision.models as models


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
    # torch_graph = scripted_model.graph
    output = scripted_model(x)
    np.testing.assert_allclose(output.detach().numpy(), output.detach().numpy())


def test_pytorch_import_export_trace(traced_model):
    x = torch.randn(1, 3, 224, 224, requires_grad=False)
    # torch_graph = traced_model.graph
    output = traced_model(x)
    np.testing.assert_allclose(output.detach().numpy(), output.detach().numpy())
