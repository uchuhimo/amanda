import tensorflow as tf
import torch

import amanda
from examples.pruning.vector_wise_sparsity import create_mask


class TypeMapping(amanda.Mapping):
    def __init__(self):
        self.register_rule(
            source="amanda/tensorflow", target="pruning", func=self.tf_type
        )
        self.register_rule(
            source="amanda/pytorch", target="pruning", func=self.torch_type
        )

    def tf_type(self, context: amanda.OpContext):
        op = context.get_op()
        context["type"] = op.type.lower()
        if op.type == "MatMul":
            context["type"] = "linear"
        if not context.is_forward():
            backward_op = context.get_backward_op()
            if op.type == "Conv2D" and backward_op.type == "Conv2DBackpropFilter":
                context["type"] = "conv2d_backward"
            if op.type == "MatMul" and backward_op.type == "MatMul":
                context["type"] = "linear_backward"

    def torch_type(self, context: amanda.OpContext):
        op = context.get_op()
        context["type"] = op.__name__
        if not context.is_forward():
            backward_op = context.get_backward_op()
            if (
                op.__name__ == "conv2d"
                and backward_op.__name__ == "CudnnConvolutionBackward"
            ):
                context["type"] = "conv2d_backward"
            if op.__name__ == "linear" and backward_op.__name__ == "AddmmBackward":
                context["type"] = "linear_backward"


class GetShapeMapping(amanda.Mapping):
    def __init__(self):
        self.register_rule(
            source="amanda/tensorflow", target="pruning", func=self.tf_get_shape
        )
        self.register_rule(
            source="amanda/pytorch", target="pruning", func=self.torch_get_shape
        )

    def tf_get_shape(self, context: amanda.OpContext):
        context["get_shape"] = lambda tensor: tensor.shape.as_list()

    def torch_get_shape(self, context: amanda.OpContext):
        context["get_shape"] = lambda tensor: tensor.shape


class GetMaskMapping(amanda.Mapping):
    def __init__(self):
        self.register_rule(
            source="amanda/tensorflow", target="pruning", func=self.tf_get_mask
        )
        self.register_rule(
            source="amanda/pytorch", target="pruning", func=self.torch_get_mask
        )

    def tf_get_shape(self, context: amanda.OpContext):
        def get_mask(weight):
            torch_weight = torch.from_numpy(weight.eval())
            if len(torch_weight.shape) == 4:
                torch_weight = torch_weight.permute(2, 3, 0, 1)
            mask = create_mask(torch_weight)
            if len(mask.shape) == 4:
                mask = mask.permute(2, 3, 0, 1)
            return tf.convert_to_tensor(mask.cpu().numpy())

        context["get_mask"] = get_mask

    def torch_get_shape(self, context: amanda.OpContext):
        context["get_mask"] = create_mask


class PruningMapping(amanda.Mapping):
    def __init__(self):
        self.use(TypeMapping())
        self.use(GetShapeMapping())
        self.use(GetMaskMapping())


class PruningTool(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="pruning")
        self.register_mapping(PruningMapping())
        self.add_inst_for_op(self.instrumentation)
        self.add_inst_for_op(
            self.backward_instrumentation,
            backward=True,
            require_outputs=True,
        )

    def instrumentation(self, context: amanda.OpContext):
        if context["type"] not in ["conv2d", "linear"]:
            return
        weight = context.get_inputs()[1]
        if (
            context["type"] == "conv2d" and context["get_shape"](weight)[3] % 4 == 0
        ) or (context["type"] == "linear" and len(context["get_shape"](weight)) == 2):
            mask = context["get_mask"](weight)
            context["mask"] = mask
            context.insert_before_op(self.mask_forward_weight, inputs=[1], mask=mask)

    def backward_instrumentation(self, context: amanda.OpContext):
        if context["type"] not in ["conv2d_backward", "linear_backward"]:
            return
        weight_grad = context.get_grad_inputs()[0]
        if (
            context["type"] == "conv2d_backward"
            and context["get_shape"](weight_grad)[3] % 4 == 0
        ) or (
            context["type"] == "linear_backward"
            and len(context["get_shape"](weight_grad)) == 2
        ):
            mask = context["mask"]
            if context["type"] == "linear_backward":
                if context["get_shape"](weight_grad) != context["get_shape"](mask):
                    return
            context.insert_after_backward_op(
                self.mask_backward_gradient, grad_inputs=[0], mask=mask
            )

    def mask_forward_weight(self, weight, mask):
        return weight * mask

    def mask_backward_gradient(self, weight_grad, mask):
        return weight_grad * mask
