cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(amanda_pybind)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${PYTHON_INCLUDE_DIRS})

find_package(pybind11)

pybind11_add_module(amanda_pybind amanda_pybind.cpp)
target_link_libraries(amanda_pybind PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(amanda_pybind PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET amanda_pybind PROPERTY CXX_STANDARD 14)
