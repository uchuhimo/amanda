import amanda
import tensorflow as tf
import torch

from examples.pruning.vector_wise_sparsity import create_mask


class PruningTool(amanda.Tool):
    def __init__(
        self, amax=127, num_bits=8, unsigned=False, narrow_range=True, device="cuda"
    ):
        super().__init__(namespace="amanda/tensorflow")
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

        # if not isinstance(self.amax,tf.Tensor):
        #     amax = tf.Tensor([self.amax])

        min_amax = amax

        max_bound = (2.0 ** (self.num_bits - 1 + int(self.unsigned))) - 1.0
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
        if op.type not in ["Conv2D", "MatMul"]:
            return
        context.insert_before_op(self.mask_forward_weight)

    def backward_instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        backward_op = context.get_backward_op()
        if (
            op is None
            or op.type not in ["Conv2D", "MatMul"]
            or backward_op.type
            not in [
                "Conv2DBackpropFilter",
                "MatMul",
            ]
        ):
            return
        print("backward instrumentation")
        context.insert_after_backward_op(self.mask_backward_gradient)

    def mask_forward_weight(self, *inputs):
        new_inputs = [
            self._tensor_quant(tensor) if isinstance(tensor, tf.Tensor) else tensor
            for idx, tensor in enumerate(inputs)
        ]
        print("forward masking")
        return new_inputs

    def mask_backward_gradient(self, *inputs):
        new_inputs = [
            self._fake_backward(tensor) if isinstance(tensor, tf.Tensor) else tensor
            for idx, tensor in enumerate(inputs)
        ]
        print("backward masking")
        return new_inputs

    def _tensor_quant(self, inputs):
        """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
        # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
        outputs = tf.clip_by_value(
            tf.round(inputs * self.scale), self.min_bound, self.max_bound
        )

        return outputs

    def _fake_backward(self, outputs):
        b_broadcast = tf.ones(tf.shape(outputs), dtype=outputs.dtype)
        new_outputs = tf.where(tf.less(outputs, outputs), b_broadcast, outputs)
        return new_outputs
