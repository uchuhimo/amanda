import copy
from collections import OrderedDict

import pytest

from amanda import Edge, InputPort, OutputPort, create_edge, create_op
from amanda.attributes import Attributes
from amanda.graph import create_control_edge, create_graph
from amanda.namespace import default_namespace, internal_namespace
from amanda.tests.utils import test_namespace
from amanda.type import DataType, unknown_type


def test_new_op():
    op = create_op(type="type1", name="op1")
    assert op.type == "type1"
    assert op.name == "op1"
    assert op.name_to_input_port == OrderedDict(
        {"0": InputPort(op, name="0", type=unknown_type)}
    )
    assert op.name_to_output_port == OrderedDict(
        {"0": OutputPort(op, name="0", type=unknown_type)}
    )
    assert op.input_ports == [InputPort(op, name="0", type=unknown_type)]
    assert op.output_ports == [OutputPort(op, name="0", type=unknown_type)]
    assert op.graph is None
    assert op.namespace is None
    assert op.control_input_port == InputPort(
        op, name=Edge.CONTROL_PORT_NAME, type=Edge.CONTROL_PORT_TYPE
    )
    assert op.control_output_port == OutputPort(
        op, name=Edge.CONTROL_PORT_NAME, type=Edge.CONTROL_PORT_TYPE
    )
    assert isinstance(op.attrs, Attributes) and len(op.attrs) == 0


def test_new_op_without_name():
    op = create_op(type="type1")
    assert op.name.startswith("type1_")


@pytest.fixture
def input1():
    return create_op(type="in1", name="input1")


@pytest.fixture
def input2():
    return create_op(type="in2", name="input2")


@pytest.fixture
def control_input1():
    return create_op(type="cin1", name="control_input1")


@pytest.fixture
def control_input2():
    return create_op(type="cin2", name="control_input2")


class DType(DataType):
    def __init__(self, dtype: str):
        super().__init__(
            namespace=test_namespace(), name="dtype", attrs=Attributes(dtype=dtype)
        )

    def __eq__(self, other):
        return isinstance(other, DType) and self.attrs["dtype"] == other.attrs["dtype"]


@pytest.fixture
def simple_op(input1, input2, control_input1, control_input2):
    op = create_op(
        type="Conv2d",
        name="test",
        namespace=default_namespace(),
        attrs=dict(kernel_shape=(3, 3)),
        inputs=["in1", "in2"],
        outputs=OrderedDict(out1=DType("int32"), out2=unknown_type),
    )
    create_graph(
        ops=[op, input1, input2, control_input1, control_input2],
        edges=[
            create_edge(input1.output_port(0), op.input_port("in1")),
            create_edge(input2.output_port(0), op.input_port("in2")),
            create_control_edge(control_input1, op),
            create_control_edge(control_input2, op),
        ],
    )
    return op


def test_new_op_with_args(input1, input2, control_input1, control_input2, simple_op):
    assert simple_op.name_to_input_port == OrderedDict(
        in1=input1.output_port(0).out_edges[0].dst,
        in2=input2.output_port(0).out_edges[0].dst,
    )
    assert simple_op.input_ports == [
        input1.output_port(0).out_edges[0].dst,
        input2.output_port(0).out_edges[0].dst,
    ]
    assert simple_op.output_port("out1").name == "out1"
    assert simple_op.output_port("out1").type == DType("int32")
    assert simple_op.output_port("out2").name == "out2"
    assert simple_op.output_port("out2").type == unknown_type
    assert simple_op.control_dependencies == [control_input1, control_input2]
    assert simple_op.name == "test"
    assert simple_op.type == "Conv2d"
    assert simple_op.attrs == Attributes(kernel_shape=(3, 3))


def test_input_ops(simple_op, input1, input2):
    assert simple_op.in_ops == [input1, input2]


def test_output_port(simple_op):
    output1 = simple_op.output_port(0)
    assert output1.op == simple_op and output1.name == "out1"
    output2 = simple_op.output_port(1)
    assert output2.op == simple_op and output2.name == "out2"


def test_input_port(simple_op):
    input1 = simple_op.input_port(0)
    assert input1.op == simple_op and input1.name == "in1"
    input2 = simple_op.input_port(1)
    assert input2.op == simple_op and input2.name == "in2"


@pytest.mark.xfail(raises=IndexError)
def test_input_port_with_negative_index(simple_op):
    simple_op.input_port(-1)


@pytest.mark.xfail(raises=IndexError)
def test_input_port_with_too_large_index(simple_op):
    simple_op.input_port(2)


def test_get_name(simple_op):
    assert simple_op.name == "test"


def test_set_name(simple_op):
    simple_op.name = "new_test"
    assert simple_op.name == "new_test"


def test_get_type(simple_op):
    assert simple_op.type == "Conv2d"


def test_set_type(simple_op):
    simple_op.type = "Conv3d"
    assert simple_op.type == "Conv3d"


def test_get_namespace(simple_op):
    assert simple_op.namespace == default_namespace()


def test_set_namespace(simple_op):
    simple_op.namespace = internal_namespace()
    assert simple_op.namespace == internal_namespace()


def test_copy_op(simple_op):
    op = simple_op
    op.attrs["mutable"] = []
    new_op = copy.copy(op)
    new_op.attrs["test"] = True
    assert "test" in new_op.attrs and "test" not in op.attrs
    new_op.attrs["mutable"].append(1)
    assert new_op.attrs["mutable"] == [1] and op.attrs["mutable"] == [1]


def test_deepcopy_op(simple_op):
    op = simple_op
    op.attrs["mutable"] = []
    new_op = copy.deepcopy(op)
    new_op.attrs["test"] = True
    assert "test" in new_op.attrs and "test" not in op.attrs
    new_op.attrs["mutable"].append(1)
    assert new_op.attrs["mutable"] == [1] and op.attrs["mutable"] == []
