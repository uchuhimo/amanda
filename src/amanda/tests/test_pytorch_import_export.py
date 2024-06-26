from functools import partial

import numpy as np
import pytest
import torch
import torch.jit
import torchvision.models as models
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.quantization.mobilenetv2 import QuantizableMobileNetV2

import amanda
from amanda.cache import cache_disabled, is_cache_enabled
from amanda.conversion.pytorch import (
    export_to_graph,
    export_to_module,
    import_from_graph,
    import_from_module,
)
from amanda.conversion.pytorch_updater import get_cache_size
from amanda.io.file import root_dir


@pytest.fixture(
    scope="module",
    params=[
        models.resnet18,
        # models.inception_v3,
        pytest.param(models.resnet50, marks=pytest.mark.slow),
        pytest.param(models.inception_v3, marks=pytest.mark.slow),
        pytest.param(models.alexnet, marks=pytest.mark.slow),
        pytest.param(models.vgg11, marks=pytest.mark.slow),
        pytest.param(models.vgg11_bn, marks=pytest.mark.slow),
        pytest.param(models.squeezenet1_1, marks=pytest.mark.slow),
        pytest.param(models.shufflenet_v2_x0_5, marks=pytest.mark.slow),
        pytest.param(models.mobilenet_v2, marks=pytest.mark.slow),
        pytest.param(models.mnasnet0_5, marks=pytest.mark.slow),
        # waiting for PyTorch to support JIT
        pytest.param(
            partial(models.detection.maskrcnn_resnet50_fpn, pretrained_backbone=False),
            id="maskrcnn_resnet50_fpn",
            marks=pytest.mark.slow,
        ),
        # pytest.param(
        #     partial(models.quantization.mobilenet_v2, quantize=True),
        #     id="quant_mobilenet_v2"
        # ),
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


class TestTool(amanda.Tool):
    def __init__(self):
        super(TestTool, self).__init__(namespace="amanda/pytorch")
        self.register_event(amanda.event.before_op_executed, self.test)

    def test(self, context: amanda.EventContext):
        op = context["op"]
        print(op.type)


def test_pytorch_with_hook(model_and_input):
    model, x = model_and_input
    tool = TestTool()
    amanda.apply(model, tool)
    model(x)


class NewTestTool(amanda.Tool):
    def __init__(self):
        super(NewTestTool, self).__init__(namespace="amanda/pytorch")
        # self.register_event(amanda.event.before_op_executed, self.test)
        # self.register_event(amanda.event.before_op_executed, self.test)
        # self.register_event(amanda.event.before_backward_op_executed, self.test)

    def test(self, context: amanda.EventContext):
        op = context["op"]
        print(op)

    def pruning_weight(self, context):
        def threshold_tensor(t: torch.Tensor, thresh=0, value=0) -> None:
            with torch.no_grad():
                t[t < thresh] = value

        if "conv" in context["op"].__name__:
            # input
            # threshold_tensor(context["args"][0], 0.5, 0)
            # weight
            threshold_tensor(context["args"][1], 0.5, 0)
            # bias
            threshold_tensor(context["args"][2], 0.5, 0)


def test_pytorch_with_new_hook(model_and_input):
    model, x = model_and_input
    tool = NewTestTool()

    with amanda.tool.apply(tool):
        y = model(x)
        y.backward(torch.zeros(y.shape))


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


def get_output(output, model):
    if isinstance(model, GeneralizedRCNN):
        return output[0]["scores"]
    # elif isinstance(model, models.Inception3):
    #     return model, torch.randn(1, 3, 299, 299)
    else:
        return output


class TestCacheTool(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="amanda/pytorch")
        self.cached_counter = 0
        self.counter = 0
        self.add_inst_for_op(self.insert_count)
        self.add_inst_for_op(self.insert_count_bw, backward=True)

    def insert_count(self, context):
        # print(context.get_op().__name__, context.get_op_id())
        self.cached_counter += 1
        context.insert_before_op(self.count_op, name=context.get_op().__name__)

    def insert_count_bw(self, context):
        # print(
        #     "bw",
        #     context.get_backward_op().__name__,
        #     context.get_backward_op_id(),
        # )
        self.cached_counter += 1
        context.insert_before_backward_op(
            self.count_op, name=context.get_backward_op().__name__
        )

    def count_op(self, *inputs, name):
        # print(name)
        self.counter += 1


def test_pytorch_cache_enabled(model_and_input):
    from amanda.conversion import pytorch_updater

    # linear = torch.nn.Linear(227, 128, bias=False)
    # x = torch.rand(3, 9, 227, 227)
    model, x = model_and_input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )

    tool = TestCacheTool()
    with amanda.tool.apply(tool):
        try:
            pytorch_updater._debug_cache = True
            assert is_cache_enabled()
            # model = model.to(device)
            if isinstance(model, GeneralizedRCNN):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            output = get_output(model(x), model)
            if not isinstance(model, QuantizableMobileNetV2):
                loss = criterion(output, torch.ones_like(output))
                optimizer.zero_grad()
                # output.backward(torch.ones_like(output))
                loss.backward()
                optimizer.step()

            output = get_output(model(x), model)
            if not isinstance(model, QuantizableMobileNetV2):
                loss = criterion(output, torch.ones_like(output))
                optimizer.zero_grad()
                # output.backward(torch.ones_like(output))
                loss.backward()
                optimizer.step()

            # y = linear(x)
            # y.backward(torch.ones_like(y))
            counter = tool.counter
            cached_counter = tool.cached_counter
            cache_num = get_cache_size()
            assert cached_counter == cache_num
            # if not isinstance(model, GeneralizedRCNN):
            #     assert counter == cache_num
            assert counter >= cache_num
            pytorch_updater._should_hit = True
            output = get_output(model(x), model)
            if not isinstance(model, QuantizableMobileNetV2):
                loss = criterion(output, torch.ones_like(output))
                optimizer.zero_grad()
                # output.backward(torch.ones_like(output))
                loss.backward()
                optimizer.step()
            # y = linear(x)
            # y.backward(torch.ones_like(y))
            assert tool.cached_counter == cached_counter
            assert tool.cached_counter == get_cache_size()
            assert tool.counter <= counter * 2
        finally:
            pytorch_updater._debug_cache = False
            pytorch_updater._should_hit = False


def test_pytorch_cache_disabled(model_and_input):
    # linear = torch.nn.Linear(227, 128, bias=False)
    # x = torch.rand(3, 9, 227, 227)
    model, x = model_and_input

    tool = TestCacheTool()
    with amanda.tool.apply(tool), cache_disabled():
        assert not is_cache_enabled()
        output = get_output(model(x), model)
        if not isinstance(model, QuantizableMobileNetV2):
            output.backward(torch.ones_like(output))
        # y = linear(x)
        # y.backward(torch.ones_like(y))
        counter = tool.counter
        cached_counter = tool.cached_counter
        cache_num = get_cache_size()
        assert cache_num == 0
        assert counter == cached_counter
        output = get_output(model(x), model)
        if not isinstance(model, QuantizableMobileNetV2):
            output.backward(torch.ones_like(output))
        # y = linear(x)
        # y.backward(torch.ones_like(y))
        assert get_cache_size() == 0
        assert tool.cached_counter == cached_counter * 2
        assert tool.counter == counter * 2


def test_pytorch_with_forward_backward_matching():
    """
    forward backward op and subgraph matching
    tested case is forward op and backward subgraph caused by broadcasting
    """

    class NewTestTool(amanda.Tool):
        def __init__(self):
            super(NewTestTool, self).__init__(namespace="amanda/pytorch")
            self.register_event(amanda.event.before_op_executed, self.print_name)
            self.register_event(
                amanda.event.after_backward_op_executed, self.print_name_bw
            )

        def print_name(self, context):
            print(context["op"].__name__)

        def print_name_bw(self, context):
            print(context["bw_op"])

    linear = torch.nn.Linear(227, 128, bias=True)
    x = torch.rand(3, 9, 227, 227)

    with amanda.tool.apply(NewTestTool()):
        y = linear(x)
        y.backward(torch.ones_like(y))
