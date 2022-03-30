cmake_minimum_required(VERSION 2.8)
project(Autorange)
find_package(CUDA REQUIRED)

include_directories("/usr/local/cuda-11.1/include")
# include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda-11.1/lib64")
# link_directories("/usr/local/cuda/lib64")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")

# Pass options to NVCC
list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
list(APPEND CUDA_NVCC_FLAGS "-v")

cuda_add_executable(autorange
    extensions.cpp
    autorange_counter.cpp
    counter.cpp
    set_metrics.cpp
    autorange_test.cpp
    kernel.cu
)
target_link_libraries(autorange cudart)
target_link_libraries(autorange cuda)
target_link_libraries(autorange nvperf_host)
target_link_libraries(autorange cupti)
