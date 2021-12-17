from typing import Dict

import amanda
import tensorflow as tf
import torch

from examples.pruning.vector_wise_sparsity import create_mask


class PruningTool(amanda.Tool):
    def __init__(self, disabled: bool = False, prune_matmul: bool = True):
        super().__init__(namespace="amanda/tensorflow")
        self.depends_on(
            amanda.tools.FilterOpTool(filter_fn=self.filter_fn),
            amanda.tools.EagerContextTool(),
        )
        self.register_event(amanda.event.before_op_executed, self.mask_forward_weight)
        self.register_event(
            amanda.event.after_backward_op_executed, self.mask_backward_gradient
        )
        self.masks: Dict[str, tf.Tensor] = {}
        self.disabled = disabled
        self.prune_matmul = prune_matmul

    def filter_fn(self, event, context):
        op = context["op"]
        if op.type not in [
            "Conv2D",
            "MatMul",
        ]:
            return False
        if not self.prune_matmul and op.type == "MatMul":
            return False
        if event == amanda.event.before_op_added:
            return True
        elif event == amanda.event.after_backward_op_added:
            backward_op = context["backward_op"]
            if backward_op.type in [
                "Conv2DBackpropFilter",
                "MatMul",
            ]:
                return True
        return False

    def mask_forward_weight(self, context: amanda.EventContext):
        if self.disabled:
            return
        op = context["op"]
        if op.type not in ["Conv2D", "MatMul"]:
            return
        weight = context["inputs"][1]
        if op.name not in self.masks:
            # print(op.name, "weight", weight.shape, weight.device)
            torch_weight = torch.from_numpy(weight.numpy())
            if len(torch_weight.shape) == 4:
                torch_weight = torch_weight.permute(2, 3, 0, 1)
            mask = create_mask(torch_weight)
            if len(mask.shape) == 4:
                mask = mask.permute(2, 3, 0, 1)
            with tf.device(weight.device):
                self.masks[op.name] = tf.convert_to_tensor(mask.cpu().numpy())
            # print(
            #     op.name, "mask", self.masks[op.name].shape, self.masks[op.name].device
            # )
        # print(op.name, "weight", weight.device)
        context["inputs"][1] = weight * self.masks[op.name]

    def mask_backward_gradient(self, context: amanda.EventContext):
        if self.disabled:
            return
        op = context["op"]
        backward_op = context["backward_op"]
        if op.type not in ["Conv2D", "MatMul"] or backward_op.type not in [
            "Conv2DBackpropFilter",
            "MatMul",
        ]:
            return
        weight_grad = context["grad_inputs"][0]
        if backward_op.type == "MatMul":
            if weight_grad.shape != self.masks[op.name].shape:
                return
        # print(op.name, backward_op.name, "weight_grad", weight_grad.device)
        context["grad_inputs"][0] = weight_grad * self.masks[op.name]
