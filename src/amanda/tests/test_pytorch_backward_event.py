import amanda
import torch
import torchvision


def test_naive_backward_func_hook():
    class CallbackCnt:
        def __init__(self):
            self.registration_cnt = 0
            self.trigger_cnt = 0
            self.hooked_func = list()

        def registration_callback(self):
            self.registration_cnt += 1

        def trigger_callback(self):
            self.trigger_cnt += 1

        def add_hook(self, grad_fn):
            def _add_hook(grad_fn):
                if grad_fn and grad_fn not in self.hooked_func:
                    if grad_fn.__class__.__name__ == "AccumulateGrad":
                        return
                    grad_fn.register_hook(
                        lambda in_tensor, out_tensor: self.trigger_callback()
                    )
                    self.registration_callback()
                    self.hooked_func.append(grad_fn)
                    for next_grad_fn, input_pos in grad_fn.next_functions:
                        _add_hook(next_grad_fn)
                else:
                    return

            _add_hook(grad_fn)

    cnt = CallbackCnt()

    model = torchvision.models.resnet50()
    x = torch.rand((2, 3, 227, 227))
    y = model(x)

    cnt.add_hook(y.grad_fn)

    y.backward(torch.rand_like(y))

    print(f"registrition: {cnt.registration_cnt} trigger: {cnt.trigger_cnt}")
    assert cnt.registration_cnt == cnt.trigger_cnt


def test_amanda_backward_func_hook():
    class TestTool(amanda.Tool):
        def __init__(self):
            super(TestTool, self).__init__(namespace="amanda/testtool")
            self.register_event(
                amanda.event.before_op_executed, self.before_callback
            )
            self.register_event(
                amanda.event.after_op_executed, self.after_callback
            )
            self.register_event(
                amanda.event.before_backward_op_executed, self.before_backward_callback
            )
            self.register_event(
                amanda.event.after_backward_op_executed, self.after_backward_callback
            )

            self.before_fw_cnt = 0
            self.after_fw_cnt = 0
            self.before_cnt = 0
            self.after_cnt = 0

        def before_callback(self, context):
            self.before_fw_cnt += 1

        def after_callback(self, context):
            self.after_fw_cnt += 1

        def before_backward_callback(self, context):
            self.before_cnt += 1

        def after_backward_callback(self, context):
            self.after_cnt += 1

    model = torchvision.models.resnet50()
    x = torch.rand((2, 3, 227, 227))

    tool = TestTool()

    with amanda.conversion.pytorch_updater.apply(tool):
        y = model(x)
        y.backward(torch.rand_like(y))

    print(f"before fw: {tool.before_fw_cnt}, after fw: {tool.after_fw_cnt}")
    print(f"before: {tool.before_cnt}, after: {tool.after_cnt}")
    assert tool.before_fw_cnt == tool.after_fw_cnt
    assert tool.before_cnt <= tool.after_fw_cnt
    assert tool.before_cnt == tool.after_cnt
