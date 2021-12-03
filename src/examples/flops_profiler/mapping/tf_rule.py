import amanda
import torch


def tf_get_shape(context: amanda.OpContext):
    context["get_shape"] = lambda tensor: tensor.shape.as_list()


def tf_type(context: amanda.OpContext):
    if not context.get_op():
        return
    context["type"] = (
        tf_type_mapping[context.get_op().type]
        if context.get_op().type in tf_type_mapping
        else context.get_op().type
    )


tf_type_mapping = {
    "Mul": "mul",
    "Add": "add",
    "Sub": "add",
    "MatMul": "matmul",
    "Relu": "relu",
    "Addv2": "add",
}
