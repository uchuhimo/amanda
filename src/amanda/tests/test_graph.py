import copy

import pytest

from amanda import (
    create_control_edge,
    create_control_input_port,
    create_control_output_port,
    create_edge,
    create_op,
)
from amanda.exception import IrremovableOpError
from amanda.graph import Op, create_graph
from amanda.namespace import default_namespace, internal_namespace


def test_new_graph():
    graph = create_graph()
    assert len(graph.ops) == 0


@pytest.fixture
def op1():
    return create_op(type="type1", name="op1", outputs=["0", "1", "2"])


@pytest.fixture
def op2():
    return create_op(type="type2", name="op2")


@pytest.fixture
def op3():
    return create_op(type="type3", name="op3", inputs=["0", "1"])


# graph topology:
# op1 -> op2 -> op3
#  |             ^
#  \_____________|
@pytest.fixture
def simple_graph(op1, op2, op3):
    graph = create_graph(
        namespace=default_namespace(),
        ops=[op1, op2, op3],
        edges=[
            create_edge(op1.output_port(0), op2.input_port(0)),
            create_edge(op1.output_port(2), op3.input_port(0)),
            create_edge(op2.output_port(0), op3.input_port(1)),
            create_control_edge(op1, op3),
        ],
    )
    return graph


def test_new_graph_with_arg(simple_graph, op1, op2, op3):
    assert set(simple_graph.ops) == {op1, op2, op3}


def test_add_op(simple_graph):
    op = create_op(type="type")
    simple_graph.add_op(op)
    assert op in simple_graph


@pytest.mark.xfail(raises=KeyError)
def test_add_existed_op(simple_graph, op1):
    simple_graph.add_op(op1)


def test_remove_op(simple_graph, op3):
    simple_graph.remove_op(op3)
    assert op3 not in simple_graph


@pytest.mark.xfail(raises=AssertionError)
def test_remove_non_existed_op(simple_graph):
    op = create_op(type="type")
    simple_graph.remove_op(op)


@pytest.mark.xfail(raises=IrremovableOpError)
def test_remove_irremovable_op(simple_graph, op1):
    simple_graph.remove_op(op1)


def test_contains(simple_graph, op1):
    op = create_op(type="type")
    assert op1 in simple_graph
    assert op not in simple_graph


def test_edges(simple_graph, op1, op2, op3):
    assert set(simple_graph.edges) == {
        create_edge(op1.output_port(0), op2.input_port(0)),
        create_edge(op1.output_port(2), op3.input_port(0)),
        create_edge(op2.output_port(0), op3.input_port(1)),
        create_control_edge(op1, op3),
    }


def test_data_edge(op1, op2):
    tensor = op1.output_port(2)
    input_port = op2.input_port(0)
    edge = create_edge(tensor, input_port)
    assert edge.src.op == op1
    assert edge.src == tensor
    assert edge.src.name == "2"
    assert edge.dst.op == op2
    assert edge.dst == input_port
    assert edge.dst.name == "0"
    assert not edge.is_control_edge()
    assert not edge.dst.is_control()


def test_control_edge(op1, op2):
    edge = create_control_edge(op1, op2)
    assert edge.src.op == op1
    assert edge.src == create_control_output_port(op1)
    assert edge.src.name == Op.CONTROL_PORT_NAME
    assert edge.dst.op == op2
    assert edge.dst == create_control_input_port(op2)
    assert edge.dst.name == Op.CONTROL_PORT_NAME
    assert edge.is_control_edge()
    assert edge.dst.is_control()


def test_post_order_ops(simple_graph, op1, op2, op3):
    assert simple_graph.sorted_ops == [op1, op2, op3]


def test_get_namespace(simple_graph):
    assert simple_graph.namespace == default_namespace()


def test_set_namespace(simple_graph):
    simple_graph.namespace = internal_namespace()
    assert simple_graph.namespace == internal_namespace()


def test_copy_graph(simple_graph):
    graph = simple_graph
    graph.attrs["mutable"] = []
    for op in graph.ops:
        op.attrs["mutable"] = []
    for edge in graph.edges:
        edge.attrs["mutable"] = []
    new_graph = graph.copy()
    new_graph.attrs["test"] = True
    assert "test" in new_graph.attrs and "test" not in graph.attrs
    new_graph.attrs["mutable"].append(1)
    assert new_graph.attrs["mutable"] == [1] and graph.attrs["mutable"] == [1]
    for new_op in new_graph.ops:
        new_op.attrs["test"] = True
        op = graph.get_op(new_op.name)
        assert "test" not in op.attrs and "test" in new_op.attrs
        new_op.attrs["mutable"].append(1)
        assert new_op.attrs["mutable"] == [1] and op.attrs["mutable"] == [1]
    for new_edge in new_graph.edges:
        new_edge.attrs["test"] = True
        if new_edge.is_control_edge():
            edge = graph.get_control_edge(
                graph.get_op(new_edge.src.op.name),
                graph.get_op(new_edge.dst.op.name),
            )
        else:
            edge = graph.get_edge(
                graph.get_op(new_edge.src.op.name).output_port(new_edge.src.name),
                graph.get_op(new_edge.dst.op.name).input_port(new_edge.dst.name),
            )
        assert "test" in new_edge.attrs and "test" not in edge.attrs
        new_edge.attrs["mutable"].append(1)
        assert new_edge.attrs["mutable"] == [1] and edge.attrs["mutable"] == [1]
    op = create_op(type="test_tf_copy_graph")
    new_graph.add_op(op)
    assert op in new_graph and op not in graph


def test_deepcopy_graph(simple_graph):
    graph = simple_graph
    graph.attrs["mutable"] = []
    for op in graph.ops:
        op.attrs["mutable"] = []
    for edge in graph.edges:
        edge.attrs["mutable"] = []
    new_graph = copy.deepcopy(graph)
    new_graph.attrs["test"] = True
    assert "test" in new_graph.attrs and "test" not in graph.attrs
    new_graph.attrs["mutable"].append(1)
    assert new_graph.attrs["mutable"] == [1] and graph.attrs["mutable"] == []
    for new_op in new_graph.ops:
        new_op.attrs["test"] = True
        op = graph.get_op(new_op.name)
        assert "test" not in op.attrs and "test" in new_op.attrs
        new_op.attrs["mutable"].append(1)
        assert new_op.attrs["mutable"] == [1] and op.attrs["mutable"] == []
    for new_edge in new_graph.edges:
        new_edge.attrs["test"] = True
        if new_edge.is_control_edge():
            edge = graph.get_control_edge(
                graph.get_op(new_edge.src.op.name),
                graph.get_op(new_edge.dst.op.name),
            )
        else:
            edge = graph.get_edge(
                graph.get_op(new_edge.src.op.name).output_port(new_edge.src.name),
                graph.get_op(new_edge.dst.op.name).input_port(new_edge.dst.name),
            )
        assert "test" in new_edge.attrs and "test" not in edge.attrs
        new_edge.attrs["mutable"].append(1)
        assert new_edge.attrs["mutable"] == [1] and edge.attrs["mutable"] == []
    op = create_op(type="test_tf_copy_graph")
    new_graph.add_op(op)
    assert op in new_graph and op not in graph
