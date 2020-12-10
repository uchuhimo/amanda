import numpy as np
import pytest
import torch
import torch.jit
import torchvision.models as models
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

import amanda
from amanda.conversion.pytorch import (
    export_to_graph,
    export_to_module,
    import_from_graph,
    import_from_module,
)
from amanda.io.file import root_dir


@pytest.fixture(
    scope="module",
    params=[
        models.resnet18,
        # models.resnet50,
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
def test_methods():
    torch._C._debug_set_autodiff_subgraph_inlining(True)
    print(f"_get_graph_executor_optimize: {torch._C._get_graph_executor_optimize()}")
    torch._C._set_graph_executor_optimize(False)
    print(torch._C._get_graph_executor_optimize())
    # print(torch._C._last_executed_optimized_graph())
    # print(f"_jit_get_profiling_mode: {torch._C._jit_get_profiling_mode()}")
    # print(f"_jit_get_profiling_executor: {torch._C._jit_get_profiling_executor()}")
    # print(f"_jit_get_num_profiled_runs: {torch._C._jit_get_num_profiled_runs()}")
    # print(f"_jit_get_bailout_depth: {torch._C._jit_get_bailout_depth()}")


# TODO: waiting for PyTorch's bugfix
# see https://github.com/pytorch/pytorch/issues/43948
# also see:
# https://github.com/pytorch/pytorch/issues/41674
# https://github.com/pytorch/pytorch/issues/34002
# https://github.com/pytorch/pytorch/issues/37720
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
    arch_name = type(model).__name__
    graph_path = root_dir() / "tmp" / "pytorch_graph" / arch_name / arch_name
    # torch.save(model, "model.pth")
    traced_model = torch.jit.trace(model, (x,))
    # torch.jit.save(traced_model, "traced_model.pth")
    output = traced_model(x)
    graph = import_from_module(traced_model)
    amanda.io.save_to_proto(graph, graph_path)
    graph = amanda.io.load_from_proto(graph_path)
    amanda.io.save_to_yaml(graph, graph_path)
    new_graph = amanda.io.load_from_yaml(graph_path)
    new_model = export_to_module(new_graph)
    new_output = new_model(x)
    assert_close(output, new_output, model)


@pytest.mark.skip
def test_pytorch_graph_callback(model_and_input):
    import torch.autograd.profiler as profiler

    profiler.profile(record_shapes=True)
    model, x = model_and_input
    arch_name = type(model).__name__
    graph_path = root_dir() / "tmp" / "pytorch_graph_callback" / arch_name / arch_name
    # traced_model = torch.jit.trace(model, (x,), check_trace=False)
    traced_model = torch.jit.trace(model, (x,))

    def after_optimize_callback(torch_graph):
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        graph = import_from_graph(torch_graph)
        # graph = import_from_graph(torch_graph, traced_model)
        amanda.io.save_to_yaml(graph, graph_path)
        new_graph = amanda.io.load_from_yaml(graph_path)
        new_torch_graph = export_to_graph(new_graph)
        return new_torch_graph
        # return torch_graph

    output = traced_model(x)
    traced_model.graph.after_optimize_callback = after_optimize_callback
    # torch._C._set_graph_executor_optimize(True)
    # torch._C._jit_set_profiling_executor(True)
    # torch._C._jit_set_profiling_mode(True)
    new_output = traced_model(x)
    assert_close(output, new_output, model)
