#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "listener.cpp"

PYBIND11_MODULE(listener, m) {
  m.doc() = "unit test for pytorch dispatcher listener";

  pybind11::class_<HookRegisterer>(m, "HookRegisterer")
    .def(pybind11::init<const std::function<std::string(std::string)> &>());
}