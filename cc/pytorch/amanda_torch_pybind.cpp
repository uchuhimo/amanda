#include "function_pre_hook.cpp"
#include "listener.cpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(amanda_torch_pybind, m) {
  m.doc() = "extra binding for Amanda on PyTorch";
  pybind11::class_<HookRegisterer>(m, "HookRegisterer")
      .def(pybind11::init<const std::function<std::string(std::string)> &>());
  m.def("amanda_add_pre_hook", &amanda_add_pre_hook, "add function pre hook");
  m.def("amanda_remove_pre_hook", &amanda_remove_pre_hook,
        "remove function pre hook");
  m.def("init_THPVariableClass", &init_THPVariableClass,
        "init THPVariableClass with torch.tensor");
}
