# type: ignore
import amanda.tools.debugging.graph as amanda


def map_pytorch_tensor(src_attr_value):
    output_tensors = src_attr_value
    for tensor in output_tensors:
        tensor.attrs["is_valid"] = tensor.attrs["type"].kind() == "TensorType"
    return output_tensors


rule = amanda.create_rule(
    src_op="*",
    src_attr_name="output_tensors",
    src_attr_value="*",
    dst_op="*",
    dst_attr_name="output_tensors",
    dst_attr_value=map_pytorch_tensor,
)
table = amanda.get_mapping_table("amanda/pytorch", "debugging")
table.insert_rule(rule)


def map_output_tensor(src_op, src_attr_value):
    output_tensors = src_attr_value
    output_tensors[0].attrs["type"] = src_op.input_tensors[0].attrs["type"]
    return output_tensors


rule1 = amanda.create_rule(
    src_op="store_tensor_to_file",
    src_attr_name="*",
    src_attr_value="*",
    dst_op="amanda::store_tensor_to_file",
    dst_attr_name="*",
    dst_attr_value="*",
)
rule2 = amanda.create_rule(
    src_op="amanda::store_tensor_to_file",
    src_attr_name="output_tensors",
    src_attr_value="*",
    dst_op="amanda::store_tensor_to_file",
    dst_attr_name="output_tensors",
    dst_attr_value=map_output_tensor,
)
table = amanda.get_mapping_table("debugging", "amanda/pytorch")
table.insert_rule(rule1)
table.insert_rule(rule2, index=1)


def map_tf_tensor(src_attr_value):
    output_tensors = src_attr_value
    for tensor in output_tensors:
        tensor.attrs["is_valid"] = not tensor.attrs["dtype"]._is_ref_dtype
    return output_tensors


rule = amanda.create_rule(
    src_op="*",
    src_attr_name="output_tensors",
    src_attr_value="*",
    dst_op="*",
    dst_attr_name="output_tensors",
    dst_attr_value=map_tf_tensor,
)
table = amanda.get_mapping_table("amanda/tensorflow", "debugging")
table.insert_rule(rule)

rule1 = amanda.create_rule(
    src_op="store_tensor_to_file",
    src_attr_name="*",
    src_attr_value="*",
    dst_op="StoreTensorToFile",
    dst_attr_name="*",
    dst_attr_value="*",
)
rule2 = amanda.create_rule(
    src_op="StoreTensorToFile",
    src_attr_name=None,
    src_attr_value=None,
    dst_op="StoreTensorToFile",
    dst_attr_name="name",
    dst_attr_value=lambda src_op: src_op.input_ops[0].name + src_op.output_ops[0].name,
)
rule3 = amanda.create_rule(
    src_op="StoreTensorToFile",
    src_attr_name=None,
    src_attr_value=None,
    dst_op="StoreTensorToFile",
    dst_attr_name="T",
    dst_attr_value=lambda src_op: src_op.input_tensors[0].attrs["dtype"],
)
table = amanda.get_mapping_table("debugging", "amanda/tensorflow")
table.insert_rule(rule1)
table.insert_rule(rule2, index=1)
table.insert_rule(rule3, index=2)
