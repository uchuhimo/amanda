#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "counter.h"

namespace py = pybind11;

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