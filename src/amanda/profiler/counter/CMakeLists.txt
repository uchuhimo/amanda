cmake_minimum_required(VERSION 3.15...3.22)
project(GPUValueCount CXX)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
	/usr/local/cuda/include 
	/usr/local/cuda/extras/CUPTI/include
	)

link_directories(
	/usr/local/cuda/lib64 
	/usr/local/cuda/extras/CUPTI/lib64
	)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -lpthread")

pybind11_add_module(counter counter_pybind.cpp counter.cpp autorange_counter.cpp extensions.cpp set_metrics.cpp)
target_link_libraries(counter PRIVATE cudart)
target_link_libraries(counter PRIVATE cuda)
target_link_libraries(counter PRIVATE nvperf_host)
target_link_libraries(counter PRIVATE cupti)