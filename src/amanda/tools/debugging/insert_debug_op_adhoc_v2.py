from torch._C import TensorType

import amanda
import amanda.matcher
from amanda import Namespace
from amanda.rule import register_op_rule, register_tensor_rule

debugging_namespace = Namespace("debugging")

register_tensor_rule(
    src_namespace=amanda.pytorch.pytorch_namespace(),
    dst_namespace=debugging_namespace,
    src_tensor=amanda.matcher.tensor(
        attrs={"type": amanda.matcher.var("type"), "value": 1}
    ),
    dst_tensor=amanda.matcher.tensor(
        attrs={
            "is_tensor": amanda.matcher.var("type").map(
                lambda type_attr: type_attr.kind() == "TensorType"
            ),
            "is_ref": False,
        }
    ),
)

register_tensor_rule(
    src_namespace=amanda.tensorflow.tf_namespace(),
    dst_namespace=debugging_namespace,
    src_tensor=amanda.matcher.tensor(attrs={"dtype": amanda.matcher.var("dtype")}),
    dst_tensor=amanda.matcher.tensor(
        attrs={
            "is_tensor": True,
            "is_ref": amanda.matcher.var("dtype").map(
                lambda dtype_attr: dtype_attr._is_ref_dtype
            ),
            "type": TensorType.get(),
        }
    ),
)


def extract_name(op):
    input_tensor = op.input_tensors[0]
    return f"debug/{input_tensor.op.attrs['name']}/{input_tensor.output_index}"


register_op_rule(
    src_namespace=debugging_namespace,
    dst_namespace=amanda.tensorflow.tf_namespace(),
    src_op=amanda.matcher.op(attrs={"type": "amanda::store_tensor_to_file"}),
    dst_op=amanda.matcher.op(
        attrs={
            "type": "StoreTensorToFile",
            "name": amanda.matcher.eval(extract_name),
            "T": amanda.matcher.eval(lambda op: op.input_tensors[0].attrs["dtype"]),
        }
    ),
)


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    graph = graph.to_namespace(debugging_namespace)
    for op in graph.ops:
        for tensor in op.output_tensors:
            if tensor.attrs["is_tensor"] and not tensor.attrs["is_ref"]:
                debug_op = amanda.create_op(
                    attrs={"type": "amanda::store_tensor_to_file"},
                    input_tensors=[tensor],
                    control_dependencies=[],
                    output_num=1,
                )
                debug_op.output_tensors[0].attrs["type"] = tensor.attrs["type"]

                for output_op in graph.ops:
                    for index, input_tensor in enumerate(output_op.input_tensors):
                        if tensor == input_tensor:
                            output_op.update_input_tensor(
                                index, debug_op.output_tensors[0]
                            )
                graph.add_op(debug_op)
    return graph.to_namespace(namespace)
