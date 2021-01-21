import torch

import amanda
from apex.contrib.sparsity.sparse_masklib import create_mask


class PruningTool(amanda.Tool):
    def __init__(self, mask_calculator="m4n2_1d", whitelist=None):
        super(PruningTool, self).__init__(namespace="amanda/pytorch")
        self.register_event(amanda.event.before_op_executed, self.prune_weights)

        self.masks = {}
        self.weights = {}
        self.masked_weights = {}
        self.whitelist = whitelist or [
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
        ]
        self.whitelist = [
            clazz.__module__ + "." + clazz.__name__
            for clazz in self.whitelist
        ]
        if isinstance(mask_calculator, str):

            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()

            self.calculate_mask = create_mask_from_pattern
        else:
            self.calculate_mask = mask_calculator

    def prune_weights(self, context: amanda.EventContext):
        op = context["op"]
        if op.type in self.whitelist:
            weight = op.attrs["weight"]
            name = op.name
            self.masked_weights[name] = weight
            if name not in self.masks:
                self.masks[name] = torch.ones_like(weight).bool()
                self.weights[name] = weight.detach().clone()
            else:
                self.weights[name][self.masks[name]] = weight[self.masks[name]]
            with torch.no_grad():
                weight.set_(self.weights[name] * self.masks[name])

    def mask_weights(self):
        with torch.no_grad():
            for name in self.masks.keys():
                self.masked_weights[name].set_(self.weights[name] * self.masks[name])

    def compute_sparse_masks(self):
        with torch.no_grad():
            for name in self.masks.keys():
                self.masks[name].set_(self.calculate_mask(self.weights[name]))
                self.masked_weights[name].set_(self.weights[name] * self.masks[name])

    def restore_pruned_weights(self):
        with torch.no_grad():
            for name in self.masks.keys():
                self.masks[name].fill_(1)
                self.masked_weights[name].set_(self.weights[name])
