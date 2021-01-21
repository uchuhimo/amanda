import tensorflow as tf
import torch

import amanda
from apex.contrib.sparsity.sparse_masklib import create_mask
import numpy as np

class PruningTool(amanda.Tool):
    def __init__(self, mask_calculator="m4n2_1d", whitelist=None):
        super(PruningTool, self).__init__(namespace="amanda/tensorflow")
        self.register_event(amanda.event.on_graph_loaded, self.prune_weights)
        self.inited = False
        self.masks = {}
        self.mask_vars = {}
        self.weights = {}
        self.masked_weights = {}
        self.whitelist = whitelist or [
            "MatMul"
        ]
        if isinstance(mask_calculator, str):

            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()

            self.calculate_mask = create_mask_from_pattern
        else:
            self.calculate_mask = mask_calculator


    def prune_weights(self, context: amanda.EventContext):
        def mask_weight(weight, mask, name):
            mask_var = tf.Variable(mask)
            self.mask_vars[name] = mask_var.name
            return weight * mask_var
        if self.inited:
            self.inited = True
            return
        graph = context["graph"]
        for op in graph.ops:
            if op.type in self.whitelist and not op.name.startswith("gradients/"):
                weight_variable = op.in_ops[1].in_ops[0]
                weight = weight_variable.attrs["value"]
                name = op.name
                self.weights[name] = weight
                mask = np.ones_like(weight).astype(np.bool)
                self.masks[name] = mask
                read_op_output = op.in_ops[1].output_port(0)
                graph.remove_edge(op.input_port(1).in_edges[0])
                masked_output = amanda.tensorflow.import_from_tf_func(mask_weight)(graph)(
                    read_op_output,
                    mask,
                    name,
                )
                graph.create_edge(masked_output, op.input_port(1))

    def mask_weights(self):
        ...
        # with torch.no_grad():
        #     for key in self.masks.keys():
        #         self.masked_weights[key].set_(self.weights[key] * self.masks[key])

    def compute_sparse_masks(self):
        for name in self.masks.keys():
            self.masks[name][:] = self.calculate_mask(torch.from_numpy(self.weights[name]).cuda()).cpu().numpy()

    def restore_pruned_weights(self):
        for name in self.masks.keys():
            self.masks[name].fill(1)
            # self.masked_weights[name].set_(self.weights[name])
