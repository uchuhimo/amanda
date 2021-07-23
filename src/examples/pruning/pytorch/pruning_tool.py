import torch
from loguru import logger

import amanda
from examples.pruning.vector_wise_sparsity import create_mask


class PruneTool(amanda.Tool):
    def __init__(self):
        super(PruneTool, self).__init__(namespace="amanda/pytorch")
        self.add_inst_for_op(self.instrumentation)
        self.add_inst_for_op(
            self.backward_instrumentation,
            backward=True,
            require_outputs=True,
        )

    def instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        if op.__name__ not in ["conv2d", "linear"]:
            return
        weight = context.get_inputs()[1]
        if ("conv2d" in op.__name__ and weight.shape[1] % 4 == 0) or (
            "linear" in op.__name__ and len(weight.shape) == 2
        ):
            mask = self.get_mask(weight)
            logger.debug(
                f"adding pruning op for {op.__name__} with shape: {mask.shape}"
            )
            context["mask"] = mask
            context.insert_before_op(self.mask_forward_weight, inputs=[1], mask=mask)

    def backward_instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        backward_op = context.get_backward_op()
        if (
            op.__name__ == "conv2d"
            and backward_op.__name__ == "CudnnConvolutionBackward"
        ):
            weight_grad = context.get_grad_inputs()[1]
            if weight_grad.shape[1] % 4 == 0:
                context.insert_after_backward_op(
                    self.mask_backward_gradient,
                    grad_inputs=[1],
                    mask=context["mask"],
                )
        if op.__name__ == "linear" and backward_op.__name__ == "AddmmBackward":
            weight_grad = context.get_grad_inputs()[2]
            if (
                weight_grad.shape[1] % 4 == 0
                and weight_grad.shape == context["mask"].shape
            ):
                context.insert_after_backward_op(
                    self.mask_backward_gradient,
                    grad_inputs=[2],
                    mask=context["mask"],
                )

    def mask_forward_weight(self, weight, mask):
        logger.debug(f"forward op with shape:{weight.shape} masked")
        return torch.mul(weight, mask)

    def mask_backward_gradient(self, weight_grad, mask):
        logger.debug(f"backward op with shape:{weight_grad.shape} masked")
        return torch.mul(weight_grad, mask)

    def get_mask(self, weight):
        return create_mask(weight)
