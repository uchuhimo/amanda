from typing import Dict
import torch
import amanda
from examples.pruning.vector_wise_sparsity import create_mask
import tensorflow as tf
import numpy as np


class PruningTool(amanda.Tool):
    def __init__(self, disabled: bool = False, prune_matmul: bool = True):
        super().__init__(namespace="amanda/tensorflow")
        self.register_event(
            amanda.event.before_op_added,
            self.mask_forward_weight
        )
        self.register_event(
            amanda.event.after_backward_op_added,
            self.mask_backward_gradient
        )
        self.masks: Dict[str, tf.Tensor] = {}
        self.masked_weight: Dict[str, str] = {}
        self.masked_grad: Dict[str, str] = {}
        self.disabled = disabled
        self.prune_matmul = prune_matmul

    def mask_forward_weight(self, context: amanda.EventContext):
        if self.disabled:
            return
        op = context["op"]
        if op.type not in ["Conv2D", "MatMul"]:
            return
        weight = context["inputs"][1]
        if op.name not in self.masks:
            with tf.device(weight.device):
                self.masks[op.name] = tf.convert_to_tensor(
                    np.random.normal(loc=0.0, scale=1.0, size=weight.shape.as_list()).astype(np.float32),
                )
        print(op.name, "weight", weight.device)
        if op.name not in self.masked_weight or weight.name != self.masked_weight[op.name]:
            context["inputs"][1] = weight * self.masks[op.name]
            self.masked_weight[op.name] = context["inputs"][1].name

    def mask_backward_gradient(self, context: amanda.EventContext):
        if self.disabled:
            return
        op = context["op"]
        backward_op = context["backward_op"]
        if op.type not in ["Conv2D", "MatMul"] or backward_op.type not in ["Conv2DBackpropFilter", "MatMul"]:
            return
        weight_grad = context["grad_inputs"][0]
        if backward_op.type == "MatMul":
            if weight_grad.shape != self.masks[op.name].shape:
                return
        print(op.name, backward_op.name, "weight_grad", weight_grad.device)
        if op.name not in self.masked_grad or weight_grad.name != self.masked_grad[op.name]:
            context["grad_inputs"][0] = weight_grad * self.masks[op.name]
            self.masked_grad[op.name] = context["grad_inputs"][0].name
