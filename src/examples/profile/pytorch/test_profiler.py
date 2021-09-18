import sys

import amanda
import pytest
import torch
import torchvision.models as models
from loguru import logger

from examples.common.pytorch.MOE.moe import MoE
from examples.profile.pytorch.profiler import Profiler

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.fixture(
    scope="module",
    params=[
        models.resnet18,
        models.inception_v3,
        models.resnet50,
        torch.nn.RNN,
        MoE,
        models.alexnet,
        models.vgg11,
        models.vgg11_bn,
        models.squeezenet1_1,
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
    else:
        return model, torch.randn(1, 3, 224, 224)


def test_pytorch_profile(model_and_input):
    model, x = model_and_input

    profiler = Profiler()

    with amanda.tool.apply(profiler):
        model(x)

    profiler.export_chrome_trace(
        f"tmp/profile_test/pytorch_{model.__class__.__name__}.json"
    )
