import amanda
import torch
from loguru import logger

from examples.pruning.vector_wise_sparsity import create_mask


class PruneTool(amanda.Tool):
    def __init__(
        self, amax=127, num_bits=8, unsigned=False, narrow_range=True, device="cuda"
    ):
        super(PruneTool, self).__init__(namespace="amanda/pytorch")
        self.add_inst_for_op(self.instrumentation)
        self.add_inst_for_op(
            self.backward_instrumentation,
            backward=True,
            require_outputs=True,
        )

        self.amax = amax
        self.num_bits = num_bits
        self.unsigned = unsigned
        self.narrow_range = narrow_range

        if not isinstance(self.amax, torch.Tensor):
            amax = torch.Tensor([self.amax]).to(device)

        min_amax = amax.min()

        max_bound = torch.tensor(
            (2.0 ** (self.num_bits - 1 + int(self.unsigned))) - 1.0, device=amax.device
        )
        if self.unsigned:
            min_bound = 0
        elif self.narrow_range:
            min_bound = -max_bound
        else:
            min_bound = -max_bound - 1
        scale = max_bound / amax

        epsilon = 1.0 / (1 << 24)
        if (
            min_amax <= epsilon
        ):  # Treat amax smaller than minimum representable of fp16 0
            zero_amax_mask = amax <= epsilon
            scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

        self.scale = scale
        self.min_bound = min_bound
        self.max_bound = max_bound

    def instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        if op.__name__ not in ["conv2d", "linear"]:
            return
        context.insert_before_op(self.mask_forward_weight)

    def backward_instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        backward_op = context.get_backward_op()
        if (
            op.__name__ == "conv2d"
            and backward_op.__name__ == "CudnnConvolutionBackward"
        ):
            context.insert_after_backward_op(self.mask_backward_gradient)
        if op.__name__ == "linear" and backward_op.__name__ == "AddmmBackward":
            context.insert_after_backward_op(self.mask_backward_gradient)

    @torch.enable_grad()
    def mask_forward_weight(self, *inputs):
        # logger.debug(f"forward op with shape:{weight.shape} masked")
        print("pruning tensor")
        new_inputs = [
            self._tensor_quant(tensor) if isinstance(tensor, torch.Tensor) else tensor
            for idx, tensor in enumerate(inputs)
        ]
        return new_inputs

    # @torch.enable_grad()
    def mask_backward_gradient(self, *inputs):
        # logger.debug(f"backward op with shape:{weight_grad.shape} masked")
        print("pruning backward tensor")
        new_inputs = [
            self._fake_backward(tensor) if isinstance(tensor, torch.Tensor) else tensor
            for idx, tensor in enumerate(inputs)
        ]
        return new_inputs

    def _tensor_quant(self, inputs):
        """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
        # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
        print(inputs.shape)
        outputs = torch.clamp(
            (inputs * self.scale).round_(), self.min_bound, self.max_bound
        )

        return outputs

    def _fake_backward(self, outputs):
        outputs[outputs < 0] = 0
        return outputs
