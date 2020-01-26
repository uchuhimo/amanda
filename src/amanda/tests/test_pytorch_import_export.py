from functools import partial

import numpy as np
import pytest
import torch
import torch.jit
import torchvision.models as models
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from amanda.conversion.pytorch import export_to_module, import_from_module


@pytest.fixture(
    scope="module",
    params=[
        models.resnet18,
        models.resnet50,
        models.inception_v3,
        models.alexnet,
        models.vgg11,
        models.vgg11_bn,
        models.squeezenet1_1,
        models.shufflenet_v2_x0_5,
        models.mobilenet_v2,
        models.mnasnet0_5,
        partial(models.detection.maskrcnn_resnet50_fpn, pretrained_backbone=False),
        # partial(models.quantization.mobilenet_v2, quantize=True),
    ],
)
def model_and_input(request):
    model = request.param(pretrained=False, progress=False)
    model.eval()
    if isinstance(model, GeneralizedRCNN):
        return model, [torch.rand(3, 300, 300)]
    elif isinstance(model, models.Inception3):
        return model, torch.randn(1, 3, 299, 299)
    else:
        return model, torch.randn(1, 3, 224, 224)


def assert_close(output, new_output, model):
    if isinstance(model, GeneralizedRCNN):
        outputs = output[1][0]
        new_outputs = new_output[1][0]
        for key in ["boxes", "labels", "scores", "masks"]:
            np.testing.assert_allclose(
                outputs[key].detach().numpy(), new_outputs[key].detach().numpy()
            )
    elif isinstance(model, models.Inception3) and isinstance(output, tuple):
        np.testing.assert_allclose(
            output.logits.detach().numpy(), new_output.logits.detach().numpy()
        )
    else:
        np.testing.assert_allclose(output.detach().numpy(), new_output.detach().numpy())


@pytest.mark.skip
def test_pytorch_import_export_script(model_and_input):
    model, x = model_and_input
    scripted_model = torch.jit.script(model)
    output = scripted_model(x)
    graph = import_from_module(scripted_model)
    new_model = export_to_module(graph)
    new_output = new_model(x)
    assert_close(output, new_output, model)


def test_pytorch_import_export_trace(model_and_input):
    model, x = model_and_input
    if isinstance(model, GeneralizedRCNN):
        return
    traced_model = torch.jit.trace(model, (x,))
    output = traced_model(x)
    graph = import_from_module(traced_model)
    new_model = export_to_module(graph)
    new_output = new_model(x)
    assert_close(output, new_output, model)
