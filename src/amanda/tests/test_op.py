import pytest

from amanda import Op


def test_new_op():
    op = Op()
    assert isinstance(op.input_tensors, list) and len(op.input_tensors) == 0
    assert (
        isinstance(op.control_dependencies, set) and len(op.control_dependencies) == 0
    )
    assert isinstance(op.attrs, dict) and len(op.attrs) == 0


@pytest.fixture
def input1():
    return Op()


@pytest.fixture
def input2():
    return Op()


@pytest.fixture
def control_input1():
    return Op()


@pytest.fixture
def control_input2():
    return Op()


@pytest.fixture
def simple_op(input1, input2, control_input1, control_input2):
    return Op(
        input_tensors=[input1.output_tensor(), input2.output_tensor()],
        control_dependencies=[control_input1, control_input2],
        attrs=dict(name="test", type="Conv2d"),
        output_num=2,
    )


def test_new_op_with_args(input1, input2, control_input1, control_input2, simple_op):
    assert simple_op.input_tensors == [input1.output_tensor(), input2.output_tensor()]
    assert simple_op.control_dependencies == {control_input1, control_input2}
    assert simple_op.attrs == dict(name="test", type="Conv2d")


def test_add_control_dependency(simple_op, control_input1, control_input2):
    control_input = Op()
    simple_op.add_control_dependency(control_input)
    assert simple_op.control_dependencies == {
        control_input,
        control_input1,
        control_input2,
    }


@pytest.mark.xfail(raises=AssertionError)
def test_add_existed_control_dependency(simple_op, control_input1):
    simple_op.add_control_dependency(control_input1)


def test_remove_control_op(simple_op, control_input1, control_input2):
    simple_op.remove_control_dependency(control_input1)
    assert simple_op.control_dependencies == {control_input2}


@pytest.mark.xfail(raises=AssertionError)
def test_remove_non_existed_control_dependency(simple_op):
    control_input = Op()
    simple_op.remove_control_dependency(control_input)


def test_input_ops(simple_op, input1, input2):
    assert simple_op.input_ops == [input1, input2]


def test_output_tensor(simple_op):
    output1 = simple_op.output_tensor()
    assert output1.op == simple_op and output1.output_index == 0
    output2 = simple_op.output_tensor(1)
    assert output2.op == simple_op and output2.output_index == 1


def test_input_port(simple_op):
    input1 = simple_op.input_port()
    assert input1.op == simple_op and input1.input_index == 0
    input2 = simple_op.input_port(1)
    assert input2.op == simple_op and input2.input_index == 1


@pytest.mark.xfail(raises=IndexError)
def test_input_port_with_negative_index(simple_op):
    simple_op.input_port(-1)


@pytest.mark.xfail(raises=IndexError)
def test_input_port_with_too_large_index(simple_op):
    simple_op.input_port(2)


def test_input_index(simple_op, input1, input2):
    assert simple_op.input_index(input1) == 0
    assert simple_op.input_index(input2) == 1


@pytest.mark.xfail(raises=IndexError)
def test_input_index_with_incorrect_op(simple_op):
    simple_op.input_index(simple_op)


def test_input_op(simple_op, input1, input2):
    assert simple_op.input_op(0) == input1
    assert simple_op.input_op(1) == input2


@pytest.mark.xfail(raises=IndexError)
def test_input_op_with_negative_index(simple_op):
    simple_op.input_op(-1)


@pytest.mark.xfail(raises=IndexError)
def test_input_op_with_too_large_index(simple_op):
    simple_op.input_op(2)


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
