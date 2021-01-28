import tensorflow as tf
import torch

import amanda
from apex.contrib.sparsity.sparse_masklib import create_mask
import numpy as np

class PruningTool(amanda.Tool):
    def __init__(self, mask_calculator="m4n2_1d", whitelist=None):
        super(PruningTool, self).__init__(namespace="amanda/tensorflow")
        self.mask_names = {}
        self.weight_names = {}
        self.whitelist = whitelist or [
            "MatMul"
        ]
        if isinstance(mask_calculator, str):

            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()

            self.calculate_mask = create_mask_from_pattern
        else:
            self.calculate_mask = mask_calculator

    def init_masks(self):
        self.unregister_event(amanda.event.on_graph_loaded)
        self.register_event(amanda.event.on_graph_loaded, self.mask_weights)

    def mask_weights(self, context: amanda.EventContext):
        def mask_weight(weight, mask, name):
            mask_var = tf.constant(mask, name=name)
            return weight * mask_var
        graph = context["graph"]
        for op in graph.ops:
            if op.type in self.whitelist and not op.name.startswith("gradients/"):
                name = op.name
                weight_variable = op.in_ops[1].in_ops[0]
                weight = weight_variable.attrs["value"]
                self.weight_names[name] = weight_variable.name
                self.mask_names[name] = name + "_mask"
                mask = np.ones_like(weight)
                read_op_output = op.in_ops[1].output_port(0)
                graph.remove_edge(op.input_port(1).in_edges[0])
                masked_output = amanda.tensorflow.import_from_tf_func(mask_weight)(graph)(
                    read_op_output,
                    mask,
                    self.mask_names[name],
                )
                graph.create_edge(masked_output, op.input_port(1))

    def compute_masks(self):
        self.unregister_event(amanda.event.on_graph_loaded)
        self.register_event(amanda.event.on_graph_loaded, self.update_masks)

    def update_masks(self, context: amanda.EventContext):
        graph: amanda.Graph = context["graph"]
        for op_name, weight_name in self.weight_names.items():
            weight = graph.get_op(weight_name).attrs["value"]
            updated_mask = self.calculate_mask(torch.from_numpy(weight).cuda()).cpu().numpy()
            graph.get_op(self.mask_names[op_name]).attrs["value"] = tf.make_tensor_proto(updated_mask)

    def remove_masks(self):
        self.unregister_event(amanda.event.on_graph_loaded)
        self.register_event(amanda.event.on_graph_loaded, self.recover_graph)

    def recover_graph(self, context: amanda.EventContext):
        graph: amanda.Graph = context["graph"]
        for mask_name in self.mask_names.values():
            mask_op = graph.get_op(mask_name)
            mul_op = mask_op.out_ops[0]
            read_op = mul_op.in_ops[0]
            target_port = mul_op.out_edges[0].dst
            graph.remove_edge(mul_op.in_edges[0])
            graph.remove_edge(mul_op.in_edges[1])
            graph.remove_edge(mul_op.out_edges[0])
            graph.add_edge(amanda.create_edge(read_op.output_port(0), target_port))
            graph.remove_op(mask_op)
            graph.remove_op(mul_op)
        self.weight_names.clear()
        self.mask_names.clear()
