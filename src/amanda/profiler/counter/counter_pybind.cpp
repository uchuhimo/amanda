#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "counter.h"

namespace py = pybind11;

// PYBIND11_MAKE_OPAQUE(std::vector<Counter::countData_t>);
// using list_cd = std::vector<Counter::countData_t>;

PYBIND11_MODULE(counter, m) {
	m.doc() = "counter pybind";

	// Bind the member in namespace "Counter"
	auto mCounter = m.def_submodule("Counter");
	py::enum_<Counter::count_Mode> (mCounter, "count_Mode")
		.value("OFFLINE_AND_ONLINE", Counter::OFFLINE_AND_ONLINE)
		.value("OFFLINE_ONLY", Counter::OFFLINE_ONLY)
		.value("ONLINE_ONLY", Counter::ONLINE_ONLY)
		.export_values();

	py::class_<Counter::countData_t> (mCounter, "countData_t")
		.def(py::init<>())
		.def_readwrite("rangeName", &Counter::countData_t::rangeName)
		.def_readwrite("metricName", &Counter::countData_t::metricName)
		.def_readwrite("gpuValue", &Counter::countData_t::gpuValue);
	
	// py::class_<list_cd>(m, "List_cd")
    //     .def(py::init<>())
    //     .def("pop_back", &list_cd::pop_back)
    //     .def("push_back", (void(list_cd::*)(const Counter::countData_t&)) & list_cd::push_back)
    //     .def("back", (Counter::countData_t & (list_cd::*) ()) & list_cd::back)
	// 	.def("clear", [](list_cd &v) { return v.clear(); })
    //     .def("__len__", [](const list_cd &v) { return v.size(); })
    //     .def(
    //         "__iter__",
    //         [](list_cd &v) { return py::make_iterator(v.begin(), v.end()); },
    //         py::keep_alive<0, 1>());

	// Bind the class counter
	py::class_<counter> (m, "counter")
		.def(py::init<>())
		.def(py::init<std::string>())
		.def(py::init<unsigned long>())
		.def(py::init<unsigned long, std::string>())
		.def_property("kindFlag", &counter::getKindFlag, &counter::setKindFlag)
		.def_property("filePath", &counter::getFilePath, &counter::setFilePath)
		.def_readwrite("countData", &counter::countData)
		.def("setFilePath", &counter::setFilePath)
		.def("setKindFlag", &counter::setKindFlag)
		.def("getFilePath", &counter::getFilePath)
		.def("getKindFlag", &counter::getKindFlag)
		.def("onlineAnalysisOnly", &counter::onlineAnalysisOnly)
		.def("offlineAnalysisOnly", &counter::offlineAnalysisOnly)
		.def("setCountDevice", &counter::setCountDevice)
		.def("setCountParams", &counter::setCountParams)
		.def("clearData", &counter::clearData)
		.def("startProfiling", &counter::startProfiling)
		.def("stopProfiling", &counter::stopProfiling)
		.def("testClearData", &counter::testClearData);
}