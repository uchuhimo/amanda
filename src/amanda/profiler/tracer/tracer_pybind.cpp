#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tracer.h"

namespace py = pybind11;

// PYBIND11_MAKE_OPAQUE(std::vector<Tracer::traceData_rt>);
// using list_rt = std::vector<Tracer::traceData_rt>;

PYBIND11_MODULE(tracer, m) {
    m.doc() = "tracer pybind";

    // // test_vector
    // py::class_<list_rt>(m, "List_rt")
    //     .def(py::init<>())
    //     .def("pop_back", &list_rt::pop_back)
    //     .def("push_back", (void(list_rt::*)(const Tracer::traceData_rt&)) & list_rt::push_back)
    //     .def("back", (Tracer::traceData_rt & (list_rt::*) ()) & list_rt::back)
    //     .def("__len__", [](const list_rt &v) { return v.size(); })
    //     .def(
    //         "__iter__",
    //         [](list_rt &v) { return py::make_iterator(v.begin(), v.end()); },
    //         py::keep_alive<0, 1>());	

	//Bind the member in namespace "Tracer"
    auto mTracer = m.def_submodule("Tracer");
	py::enum_<Tracer::trace_Mode> (mTracer, "trace_Mode")
		.value("OFFLINE_AND_ONLINE", Tracer::OFFLINE_AND_ONLINE)
		.value("OFFLINE_ONLY", Tracer::OFFLINE_ONLY)
		.value("ONLINE_ONLY", Tracer::ONLINE_ONLY)
		.export_values();

	py::class_<Tracer::traceData_rt> (mTracer, "traceData_rt")
		.def(py::init<>())
		.def_readwrite("startTime", &Tracer::traceData_rt::startTime)
		.def_readwrite("endTime", &Tracer::traceData_rt::endTime)
		.def_readwrite("durationTime", &Tracer::traceData_rt::durationTime)
		.def_readwrite("deviceId", &Tracer::traceData_rt::deviceId)
		.def_readwrite("contextId", &Tracer::traceData_rt::contextId)
		.def_readwrite("streamId", &Tracer::traceData_rt::streamId)
		.def_readwrite("correlationId", &Tracer::traceData_rt::correlationId)
		.def_readwrite("kind", &Tracer::traceData_rt::kind)
		.def_readwrite("name", &Tracer::traceData_rt::name);

	py::class_<Tracer::traceData_api> (mTracer, "traceData_api")
		.def(py::init<>())
		.def_readwrite("startTime", &Tracer::traceData_api::startTime)
		.def_readwrite("endTime", &Tracer::traceData_api::endTime)
		.def_readwrite("durationTime", &Tracer::traceData_api::durationTime)
		.def_readwrite("processId", &Tracer::traceData_api::processId)
		.def_readwrite("threadId", &Tracer::traceData_api::threadId)
		.def_readwrite("correlationId", &Tracer::traceData_api::correlationId)
		.def_readwrite("kind", &Tracer::traceData_api::kind)
		.def_readwrite("name", &Tracer::traceData_api::name);

	py::class_<Tracer::traceData_oh> (mTracer, "traceData_oh")
		.def(py::init<>())
		.def_readwrite("startTime", &Tracer::traceData_oh::startTime)
		.def_readwrite("endTime", &Tracer::traceData_oh::endTime)
		.def_readwrite("durationTime", &Tracer::traceData_oh::durationTime)
		.def_readwrite("objectId", &Tracer::traceData_oh::objectId)
		.def_readwrite("kind", &Tracer::traceData_oh::kind)
		.def_readwrite("overheadKind", &Tracer::traceData_oh::overheadKind)
		.def_readwrite("objectKind", &Tracer::traceData_oh::objectKind);

	// Bind the class "tracer"
	py::class_<tracer> (m, "tracer")
		.def(py::init<>())
		.def(py::init<unsigned long>())
		.def(py::init<std::string>())
		.def(py::init<unsigned long, std::string>())
		.def_property("kindFlag", &tracer::getKindFlag, &tracer::setKindFlag)
		.def_property("filePath", &tracer::getFilePath, &tracer::setFilePath)
		.def_property("dataTypeFlag", &tracer::getDataTypeFlag, &tracer::setDataTypeFlag)
		.def_readwrite("traceData_rt", &tracer::traceData_rt)
		.def_readwrite("traceData_api", &tracer::traceData_api)
		.def_readwrite("traceData_oh", &tracer::traceData_oh)
		.def("setKindFlag", &tracer::setKindFlag)
		.def("getKindFlag", &tracer::getKindFlag)
		.def("setFilePath", &tracer::setFilePath)
		.def("getFilePath", &tracer::getFilePath)
		.def("setDataTypeFlag", &tracer::setDataTypeFlag)
		.def("getDataTypeFlag", &tracer::getDataTypeFlag)
		.def("onlineAnalysisOnly", &tracer::onlineAnalysisOnly)
		.def("offlineAnalysisOnly", &tracer::offlineAnalysisOnly)
		.def("activityFluashAll", &tracer::activityFlushAll)
		.def("initTrace", &tracer::initTrace)
		.def("finishTrace", &tracer::finishTrace);

}