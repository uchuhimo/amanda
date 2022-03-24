cmake_minimum_required(VERSION 2.8)
project(ActivityTrace)
find_package(CUDA REQUIRED)

include_directories(
	/usr/local/cuda/include 
	/usr/local/cuda/extras/CUPTI/include
	)

link_directories(
	/usr/local/cuda/lib64 
	/usr/local/cuda/extras/CUPTI/lib64
	)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")

# Pass options to NVCC
list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
list(APPEND CUDA_NVCC_FLAGS "-v")

cuda_add_executable(activity_trace_async
    activity_trace_async.cpp
    kernel.cu
)
target_link_libraries(activity_trace_async cudart)
target_link_libraries(activity_trace_async cuda)
target_link_libraries(activity_trace_async nvperf_host)
target_link_libraries(activity_trace_async cupti)