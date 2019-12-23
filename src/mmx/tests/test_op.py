import pytest

from mmx import Op


def test_new_op():
    op = Op()
    assert isinstance(op.inputs, list) and len(op.inputs) == 0
    assert isinstance(op.control_inputs, set) and len(op.control_inputs) == 0
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
        inputs=[input1.output(), input2.output()],
        control_inputs=[control_input1, control_input2],
        attrs=dict(name="test", type="Conv2d"),
    )


def test_new_op_with_args(input1, input2, control_input1, control_input2, simple_op):
    assert simple_op.inputs == [input1.output(), input2.output()]
    assert simple_op.control_inputs == {control_input1, control_input2}
    assert simple_op.attrs == dict(name="test", type="Conv2d")


def test_add_control_op(simple_op, control_input1, control_input2):
    control_input = Op()
    simple_op.add_control(control_input)
    assert simple_op.control_inputs == {control_input, control_input1, control_input2}


@pytest.mark.xfail(raises=AssertionError)
def test_add_existed_control_op(simple_op, control_input1):
    simple_op.add_control(control_input1)


def test_remove_control_op(simple_op, control_input1, control_input2):
    simple_op.remove_control(control_input1)
    assert simple_op.control_inputs == {control_input2}


@pytest.mark.xfail(raises=AssertionError)
def test_remove_non_existed_control_op(simple_op):
    control_input = Op()
    simple_op.remove_control(control_input)


def test_input_ops(simple_op, input1, input2):
    assert simple_op.input_ops == [input1, input2]


def test_output(simple_op):
    output1 = simple_op.output()
    assert output1.op == simple_op and output1.output_index == 0
    output2 = simple_op.output(1)
    assert output2.op == simple_op and output2.output_index == 1


def test_input(simple_op):
    input1 = simple_op.input()
    assert input1.op == simple_op and input1.input_index == 0
    input2 = simple_op.input(1)
    assert input2.op == simple_op and input2.input_index == 1


@pytest.mark.xfail(raises=IndexError)
def test_input_with_negative_index(simple_op):
    simple_op.input(-1)


@pytest.mark.xfail(raises=IndexError)
def test_input_with_too_large_index(simple_op):
    simple_op.input(2)


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