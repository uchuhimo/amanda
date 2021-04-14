# # partition for pybind
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(listener_lib)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package(pybind11)

include_directories()


pybind11_add_module(listener listener_pybind.cpp)
target_link_libraries(listener PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET listener PROPERTY CXX_STANDARD 14)

add_executable(listener_test listener.cpp)
target_link_libraries(listener_test PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET listener_test PROPERTY CXX_STANDARD 14)
