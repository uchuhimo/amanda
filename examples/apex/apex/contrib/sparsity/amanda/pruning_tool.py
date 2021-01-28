from functools import partial
import types
import torch

import amanda
from apex.contrib.sparsity.sparse_masklib import create_mask


class PruningTool(amanda.Tool):
    def __init__(self, mask_calculator="m4n2_1d", whitelist=None):
        super(PruningTool, self).__init__(namespace="amanda/pytorch")

        self.masks = {}
        self.unmasked_weights = {}
        self.weights = {}
        self.handles = {}
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

    def init_masks(self):
        self.register_event(amanda.event.before_subgraph_executed, self.mask_weights)

    def mask_weights(self, context: amanda.EventContext):
        module = context["subgraph"]
        if module.type in self.whitelist:
            weight = module.attrs["weight"]
            name = module.name
            if name not in self.masks:
                mask = torch.ones_like(weight).bool()
                self.masks[name] = mask
                self.weights[name] = weight
                self.unmasked_weights[name] = weight.detach().clone()
                def hook(grad):
                    with torch.no_grad():
                        masked_grad = grad * mask
                        self.unmasked_weights[name].add_(masked_grad)
                    return masked_grad
                self.handles[name] = weight.register_hook(hook)

    def compute_masks(self):
        with torch.no_grad():
            for name in self.masks.keys():
                masks = self.masks[name]
                unmasked_weights = self.unmasked_weights[name]
                weight = self.weights[name]
                masks.set_(self.calculate_mask(unmasked_weights))
                weight.set_(unmasked_weights * masks)

    def remove_masks(self):
        self.masks.clear()
        self.weights.clear()
        self.unmasked_weights.clear()
        for name in self.masks.keys():
            self.handles[name].remove()
