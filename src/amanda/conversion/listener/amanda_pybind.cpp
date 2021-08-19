#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "listener.cpp"
#include "function_pre_hook.cpp"

PYBIND11_MODULE(amanda_pybind, m) {
    m.doc() = "extra binding for amanda";

    pybind11::class_<HookRegisterer>(m, "HookRegisterer")
        .def(pybind11::init<const std::function<std::string(std::string)> &>());
    m.def("amanda_add_pre_hook", &amanda_add_pre_hook, "function to add a hook");
    m.def("amanda_remove_pre_hook", &amanda_remove_pre_hook, "function to remove hook by handle");
}
