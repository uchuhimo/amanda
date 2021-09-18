cmake_minimum_required(VERSION 3.17)
project(amanda_pybind)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(SYSTEM ${Python_INCLUDE_DIRS})

pybind11_add_module(amanda_pybind amanda_pybind.cpp)
target_link_libraries(amanda_pybind PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(amanda_pybind PRIVATE ${TORCH_LIBRARIES})
