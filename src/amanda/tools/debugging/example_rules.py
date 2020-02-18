# type: ignore
import amanda.tools.debugging.graph as amanda


def update_all_op_type(op: amanda.Op):
    op.type = "conv"


def update_conv_op_type(op: amanda.Op):
    if op.type == "conv_v1":
        op.type = "conv_v2"


def update_conv_op_name(op: amanda.Op):
    if op.type == "conv":
        op.name = "new_conv_name"


def update_single_op_name(op: amanda.Op):
    if op.name == "conv_3":
        op.name = "new_conv_3"
