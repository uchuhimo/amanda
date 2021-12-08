from typing import Dict

import amanda


def torch_get_shape(context: amanda.OpContext):
    context["get_shape"] = lambda tensor: tensor.shape


def torch_type(context: amanda.OpContext):
    context["type"] = (
        torch_type_mapping[context.get_op().__name__]
        if context.get_op().__name__ in torch_type_mapping
        else context.get_op().__name__
    )


torch_type_mapping: Dict[str, str] = dict()
