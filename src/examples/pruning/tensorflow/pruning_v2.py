from typing import Dict

import tensorflow as tf
import torch

import amanda
from examples.pruning.vector_wise_sparsity import create_mask


class PruningTool(amanda.Tool):
    def __init__(self, disabled: bool = False):
        super().__init__(namespace="amanda/tensorflow")
        self.add_inst_for_op(self.instrumentation)
        self.add_inst_for_op(
            self.backward_instrumentation,
            backward=True,
            require_outputs=True,
        )
        self.masks: Dict[str, tf.Tensor] = {}
        self.disabled = disabled

    def instrumentation(self, context: amanda.OpContext):
        if self.disabled:
            return
        op = context.get_op()
        if op.type not in ["Conv2D", "MatMul"]:
            return
        weight = context.get_inputs()[1]
        if ("Conv2D" == op.type and weight.shape.as_list()[3] % 4 == 0) or (
            "MatMul" == op.type and len(weight.shape) == 2
        ):
            mask = self.get_mask(weight, context["session"])
            context["mask"] = mask
            context.insert_before_op(self.mask_forward_weight, inputs=[1], mask=mask)

    def backward_instrumentation(self, context: amanda.OpContext):
        if self.disabled:
            return
        op = context.get_op()
        backward_op = context.get_backward_op()
        if op.type not in ["Conv2D", "MatMul"] or backward_op.type not in [
            "Conv2DBackpropFilter",
            "MatMul",
        ]:
            return
        weight_grad = context.get_grad_inputs()[0]
        if (
            "Conv2D" == op.type
            and "Conv2DBackpropFilter" == backward_op.type
            and weight_grad.shape.as_list()[3] % 4 == 0
        ) or (
            "MatMul" == op.type
            and "MatMul" == backward_op.type
            and len(weight_grad.shape) == 2
        ):
            mask = context["mask"]
            if backward_op.type == "MatMul":
                if weight_grad.shape != mask.shape:
                    return
            context.insert_after_backward_op(
                self.mask_backward_gradient, grad_inputs=[0], mask=mask
            )

    def mask_forward_weight(self, weight, mask):
        return weight * mask

    def mask_backward_gradient(self, weight_grad, mask):
        return weight_grad * mask

    def get_mask(self, weight, session):
        torch_weight = torch.from_numpy(weight.eval())
        if len(torch_weight.shape) == 4:
            torch_weight = torch_weight.permute(2, 3, 0, 1)
        mask = create_mask(torch_weight)
        if len(mask.shape) == 4:
            mask = mask.permute(2, 3, 0, 1)
        return tf.convert_to_tensor(mask.cpu().numpy())
