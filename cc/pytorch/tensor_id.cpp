#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/extension.h>

using torch::autograd::Variable;

uintptr_t tensor_id(py::object py_variable) {
  Variable *variable = &(((THPVariable *)py_variable.ptr())->cdata);
  return reinterpret_cast<uintptr_t>(variable);
}
