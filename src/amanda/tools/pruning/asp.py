# type: ignore
import types

import torch
import torchvision

from .sparse_masklib import create_mask


def eligible_modules(model, whitelist_layer_types):
    eligible_modules_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, whitelist_layer_types):
            eligible_modules_list.append((name, mod))
    return eligible_modules_list


class ASP:
    __model = None
    __optimizer = None
    __sparse_parameters = []
    __calculate_mask = None

    @classmethod
    def init_model_for_pruning(cls, model, mask_calculator="m4n2_1d"):
        assert cls.__model is None, "ASP has been initialized already."
        cls.__model = model
        whitelist = (
            [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d],
        )

        if isinstance(mask_calculator, str):

            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()

            cls.__calculate_mask = create_mask_from_pattern
        else:
            cls.__calculate_mask = mask_calculator  # user defined function

        # function to extract variables that will be sparsified.
        # idea is that you will add one of these functions for each module type that
        # can be sparsified.
        sparse_parameter_list = {
            torch.nn.Linear: ["weight"],
            torch.nn.Conv1d: ["weight"],
            torch.nn.Conv2d: ["weight"],
            torch.nn.Conv3d: ["weight"],
            torchvision.ops.misc.Conv2d: ["weight"],
        }

        # find all sparse modules, extract sparse parameters and decorate
        def add_sparse_attributes(module_name, module):
            sparse_parameters = sparse_parameter_list[type(module)]
            for p_name, p in module.named_parameters():
                if p_name in sparse_parameters and p.requires_grad:
                    mask = torch.ones_like(p).bool()
                    buffname = p_name.split(".")[-1]  # buffer names cannot contain "."
                    module.register_buffer("__%s_mma_mask" % buffname, mask)
                    pruned = torch.zeros_like(p).cpu()
                    module.register_buffer("__%s_mma_pruned_p" % buffname, pruned)
                    cls.__sparse_parameters.append(
                        (module_name, module, p_name, p, mask, pruned)
                    )

        for name, sparse_module in eligible_modules(model, tuple(whitelist)):
            add_sparse_attributes(name, sparse_module)

    @classmethod
    def init_optimizer_for_pruning(cls, optimizer):
        """Call this method to monkey patch optimizer step function so that masks
        can be applied to
        gradients and weights during training.
        You must call init_model_for_pruning(...) before calling
        init_optimizer_for_pruning(...)
        """
        assert cls.__optimizer is None, "ASP has initialized optimizer already."
        assert (
            cls.__calculate_mask is not None
        ), "Called ASP.init_optimizer_for_pruning before ASP.init_model_for_pruning."

        # store pointer to original optimizer step method
        cls.__optimizer = optimizer
        cls.__optimizer.__step = optimizer.step

        def __step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for (
                    module_name,
                    module,
                    p_name,
                    p,
                    mask,
                    pruned,
                ) in cls.__sparse_parameters:
                    if p.grad is not None:  # thx pjudd
                        p.grad.mul_(mask)
            # call original optimizer step method
            rval = opt_self.__step(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for (
                    module_name,
                    module,
                    p_name,
                    p,
                    mask,
                    pruned,
                ) in cls.__sparse_parameters:
                    p.mul_(mask)
            return rval

        cls.__optimizer.step = types.MethodType(__step, cls.__optimizer)

    @classmethod
    def compute_sparse_masks(cls):
        """Call this method to enable sparsity.
        If init(...) was called with allow_recompute_mask=False AND sparsity
        is disabled, pruned field can be None.
        """
        with torch.no_grad():
            for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                if mask.sum() < mask.numel():  # when recalculating masks
                    # restore dense parameter if allow_recompute_mask is enabled
                    assert pruned is not None, (
                        "Unable to restore dense parameter "
                        "because allow_recompute_mask == False"
                    )
                    p.add_(pruned.cuda())

                mask.set_(cls.__calculate_mask(p))

                if pruned is not None:  # stow away pruned weights to cpu
                    pruned.set_((p * (~mask)).cpu())

                p.mul_(mask)  # in-place multiplication, so pruned weights are 0-values,
                # hence checkpoint will have 0s for pruned weights
