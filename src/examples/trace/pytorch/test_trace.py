import sys

import amanda
import pytest
import torch
import torchvision.models as models
from loguru import logger

from examples.common.pytorch.MOE.moe import MoE
from examples.trace.pytorch.trace_tool import TraceTool

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.fixture(
    scope="module",
    params=[
        # models.resnet18,
        # models.inception_v3,
        # models.resnet50,
        # torch.nn.RNN,
        MoE,
        pytest.param(models.inception_v3, marks=pytest.mark.slow),
        pytest.param(models.alexnet, marks=pytest.mark.slow),
        # models.vgg11,
        pytest.param(models.vgg11_bn, marks=pytest.mark.slow),
        # models.squeezenet1_1,
        # models.shufflenet_v2_x0_5,
        pytest.param(models.mobilenet_v2, marks=pytest.mark.slow),
        # models.mnasnet0_5,
        # waiting for PyTorch to support
        # partial(models.detection.maskrcnn_resnet50_fpn, pretrained_backbone=False),
        # partial(models.quantization.mobilenet_v2, quantize=True),
    ],
)
def model_and_input(request):
    if request.param is torch.nn.RNN:
        model = request.param(
            input_size=128, hidden_size=128, num_layers=4, batch_first=False
        )
        model.eval()
        return model, torch.rand(16, 2, 128)
    if request.param is MoE:
        model = request.param(
            input_size=128, output_size=10, num_experts=4, hidden_size=128, k=2
        )
        model.eval()
        return model, torch.rand(1, 128)
    model = request.param(pretrained=False, progress=False)
    model.eval()
    if isinstance(model, models.Inception3):
        return model, torch.randn(1, 3, 299, 299)
    elif isinstance(model, torch.nn.RNN):
        return
    else:
        return model, torch.randn(1, 3, 224, 224)


def test_pytorch_trace(model_and_input):
    model, x = model_and_input

    tool = TraceTool(output_dir=f"tmp/trace_usecase/{model.__class__.__name__}.txt")

    with amanda.tool.apply(tool):
        y = model(x)
        if isinstance(model, MoE):
            y[0].backward(torch.rand_like(y[0]), retain_graph=True)
            y[1].backward(torch.rand_like(y[1]))
        elif isinstance(y, tuple):
            y[0].backward(torch.rand_like(y[0]))
        else:
            y.backward(torch.rand_like(y))
