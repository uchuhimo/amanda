import torch

import amanda
from amanda.tools.pruning.sparse_masklib import create_mask


class PruningTool(amanda.Tool):
    def __init__(self, mask_calculator="m4n2_1d"):
        self.register_event(amanda.event.before_op_executed, self.prune_weights)

        self.masks = {}
        self.weights = {}
        self.whitelist = [
            "torch.nn.Linear",
            "torch.nn.Conv1d",
            "torch.nn.Conv2d",
            "torch.nn.Conv3d",
        ]
        if isinstance(mask_calculator, str):

            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()

            self.calculate_mask = create_mask_from_pattern
        else:
            self.calculate_mask = mask_calculator

    def prune_weights(self, context: amanda.EventContext):
        op = context["op"]
        if op.type in self.whitelist and op.name not in self.masks:
            weight = op.attrs["weight"]
            self.masks[op.name] = torch.ones_like(weight).bool()
            self.weights[op.name] = weight
            mask = self.masks[op.name]
            op.attrs["weight"] = weight * mask
            context["new_op"] = op
            context.trigger(amanda.event.update_op)

    def compute_sparse_masks(self):
        for name, mask in self.masks.items():
            self.masks[name].set_(self.calculate_mask(self.weights[name]))
