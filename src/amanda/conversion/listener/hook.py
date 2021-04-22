import torch

from amanda.conversion.listener.build.listener import HookRegisterer


def parse_opname_to_pyname(name: str) -> str:
    def remove_namespace(name: str) -> str:
        pos = name.find("::")
        if not pos == -1:
            return name[pos + 2 :]
        else:
            return name

    name = remove_namespace(name)

    if "." in name:
        pyname, overload_name = name.split(".", 1)
    else:
        pyname = name
        # overload_name = ""

    return pyname


def get_python_op(name: str):
    TORCH_OP_MOUNT_POS = [
        torch._C._TensorBase,  # python_variable_methods.cpp/variable_methods
        torch._C._VariableFunctionsClass,  # python_torch_functions.cpp/torch_functions
        torch._C._nn,  # python_nn_functions.cpp/nn_functions
        torch._C._fft,  # python_fft_functions.cpp/fft_functions
        torch._C._linalg,  # python_linalg_functions.cpp/linalg_functions
        torch._C._VariableFunctions,
    ]

    py_name = parse_opname_to_pyname(name)
    for submodule in TORCH_OP_MOUNT_POS:
        if py_name in dir(submodule):
            print(f"{py_name} exists in submodule {submodule.__name__}")
            return py_name
    print(f"{name} does not exits")
    return py_name


def test_parse_opname_to_pyname():
    opname = "aten::_foreach_add.ScalarList"
    print(opname, parse_opname_to_pyname(opname))
    print(opname, parse_opname_to_pyname(parse_opname_to_pyname(opname)))


def test_hook_listener():
    HookRegisterer(get_python_op)


if __name__ == "__main__":
    # print(dir(listener))

    # test_parse_opname_to_pyname()

    test_hook_listener()
