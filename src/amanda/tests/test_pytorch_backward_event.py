import torch
import torchvision

import amanda
from amanda.conversion.pytorch_updater import _grad_fns
from amanda.io.file import ensure_dir


def test_naive_backward_func_hook(model=None):
    class CallbackCnt:
        def __init__(self):
            self.registration_cnt = 0
            self.trigger_cnt = 0
            self.hooked_func = list()

            self.output_file = open(
                ensure_dir("tmp/trace_resnet50/naive_trace.txt"), "w"
            )

        def registration_callback(self):
            self.registration_cnt += 1

        def trigger_callback(self, op, handle):
            self.trigger_cnt += 1
            self.output_file.write(f"{op.__name__}\n")
            handle.remove()

        def add_hook(self, grad_fn):
            def _add_hook(grad_fn):
                if grad_fn and grad_fn not in self.hooked_func:
                    handle = grad_fn.register_hook(
                        lambda in_tensor, out_tensor: self.trigger_callback(
                            grad_fn.__class__,
                            handle,
                        )
                    )
                    self.registration_callback()
                    self.hooked_func.append(grad_fn)
                    for next_grad_fn, input_pos in grad_fn.next_functions:
                        _add_hook(next_grad_fn)
                else:
                    return

            _add_hook(grad_fn)

    cnt = CallbackCnt()

    if not model:
        model = torchvision.models.resnet50()
    x = torch.rand((2, 3, 227, 227))
    y = model(x)

    cnt.add_hook(y.grad_fn)

    y.backward(torch.rand_like(y))

    print(f"registration: {cnt.registration_cnt} trigger: {cnt.trigger_cnt}")
    assert cnt.registration_cnt == cnt.trigger_cnt

    return cnt.trigger_cnt


def test_amanda_backward_func_hook(model=None):
    class TestTool(amanda.Tool):
        def __init__(self):
            super(TestTool, self).__init__(namespace="amanda/testtool")
            self.add_inst_for_op(self.before_callback)
            self.add_inst_for_op(self.after_callback, require_outputs=True)
            self.add_inst_for_op(self.before_backward_callback, backward=True)
            self.add_inst_for_op(
                self.after_backward_callback, backward=True, require_outputs=True
            )

            self.before_fw_cnt = 0
            self.after_fw_cnt = 0
            self.before_cnt = 0
            self.after_cnt = 0

            self.output_file = open(
                ensure_dir("tmp/trace_resnet50/amanda_trace.txt"), "w"
            )

        def before_callback(self, context):
            self.before_fw_cnt += 1

        def after_callback(self, context):
            self.after_fw_cnt += 1

        def before_backward_callback(self, context):
            self.before_cnt += 1

        def after_backward_callback(self, context):
            self.after_cnt += 1
            self.output_file.write(f'{context["backward_op"].__name__}\n')
            print(context["backward_op"].__name__)

    if not model:
        model = torchvision.models.resnet50()
    x = torch.rand((2, 3, 227, 227))

    tool = TestTool()

    with amanda.tool.apply(tool):
        y = model(x)
        y.backward(torch.rand_like(y))

    print(f"before fw: {tool.before_fw_cnt}, after fw: {tool.after_fw_cnt}")
    print(f"after fw: {tool.after_fw_cnt}, before bw: {tool.before_cnt}")
    print(f"before bw: {tool.before_cnt}, after bw: {tool.after_cnt}")
    assert tool.before_fw_cnt == tool.after_fw_cnt
    assert tool.before_cnt <= tool.after_fw_cnt
    assert tool.before_cnt <= tool.after_cnt
    assert len(_grad_fns) == 0

    return tool.after_cnt


def test_amanda_backward_graph_trace():

    model = torchvision.models.resnet50()

    baseline_cnt = test_naive_backward_func_hook(model)
    amanda_cnt = test_amanda_backward_func_hook(model)

    assert baseline_cnt == amanda_cnt
