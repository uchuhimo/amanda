import pytest

from mmx import ControlEdge, DataEdge, Graph, InputPort, Op, OutputPort
from mmx.exception import IrremovableOpError


def test_new_graph():
    graph = Graph()
    assert isinstance(graph.ops, set) and len(graph.ops) == 0


@pytest.fixture
def op1():
    return Op(attrs=dict(name="op1"))


@pytest.fixture
def op2(op1):
    return Op(inputs=[op1.output()], attrs=dict(name="op2"))


@pytest.fixture
def op3(op1, op2):
    return Op(
        inputs=[op1.output(2), op2.output()],
        control_inputs=[op1],
        attrs=dict(name="op3"),
    )


# graph topology:
# op1 -> op2 -> op3
#  |             ^
#  \_____________|
@pytest.fixture
def simple_graph(op1, op2, op3):
    return Graph(ops=[op1, op2, op3])


def test_new_graph_with_arg(simple_graph, op1, op2, op3):
    assert simple_graph.ops == {op1, op2, op3}


def test_add_op(simple_graph):
    op = Op()
    simple_graph.add(op)
    assert op in simple_graph.ops


@pytest.mark.xfail(raises=AssertionError)
def test_add_existed_op(simple_graph, op1):
    simple_graph.add(op1)


def test_remove_op(simple_graph, op3):
    simple_graph.remove(op3)
    assert op3 not in simple_graph.ops


@pytest.mark.xfail(raises=AssertionError)
def test_remove_non_existed_op(simple_graph):
    op = Op()
    simple_graph.remove(op)


@pytest.mark.xfail(raises=IrremovableOpError)
def test_remove_irremovable_op(simple_graph, op1):
    simple_graph.remove(op1)


def test_contains(simple_graph, op1):
    op = Op()
    assert op1 in simple_graph
    assert op not in simple_graph


@pytest.fixture
def sub_graph(op2, op3):
    return Graph(ops=[op2, op3])


def test_inputs(sub_graph, op1):
    assert set(sub_graph.inputs) == {op1.output(0), op1.output(2)}


def test_control_inputs(sub_graph, op1):
    assert sub_graph.control_inputs == {op1}


def test_set_attr(simple_graph, op1, op2, op3):
    simple_graph.set_attr("type", "Conv2d")
    assert op1.type == op2.type == op3.type == "Conv2d"


def test_edges(simple_graph, sub_graph, op1, op2, op3):
    assert set(simple_graph.edges) == {
        DataEdge(op1.output(), op2.input(0)),
        DataEdge(op1.output(2), op3.input(0)),
        DataEdge(op2.output(), op3.input(1)),
        ControlEdge(op1, op3),
    }
    assert set(sub_graph.edges) == {
        DataEdge(op2.output(), op3.input(1)),
    }


def test_data_edge(op1, op2):
    output_port = op1.output(2)
    input_port = op2.input()
    edge = DataEdge(output_port, input_port)
    assert edge.src == op1
    assert edge.src_port == output_port
    assert edge.src_output_index == 2
    assert edge.dst == op2
    assert edge.dst_port == input_port
    assert edge.dst_input_index == 0


def test_control_edge(op1, op2):
    edge = ControlEdge(op1, op2)
    assert edge.src == op1
    assert edge.src_port == OutputPort(op1, ControlEdge.CONTROL_EDGE_INDEX)
    assert edge.src_output_index == ControlEdge.CONTROL_EDGE_INDEX
    assert edge.dst == op2
    assert edge.dst_port == InputPort(op2, ControlEdge.CONTROL_EDGE_INDEX)
    assert edge.dst_input_index == ControlEdge.CONTROL_EDGE_INDEX


def test_post_order_ops(simple_graph, op1, op2, op3):
    assert simple_graph.post_order_ops == [op1, op2, op3]
