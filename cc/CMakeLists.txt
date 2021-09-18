cmake_minimum_required(VERSION 3.17)
project(Amanda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake/modules)
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)

find_package(Python COMPONENTS Interpreter Development NumPy)
find_package(TensorFlow REQUIRED)
# find_package(CUDA 9 REQUIRED)

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

include_directories(SYSTEM ${Python_INCLUDE_DIRS})
include_directories(SYSTEM ${Python_NumPy_INCLUDE_DIRS})
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})
include_directories(SYSTEM "kernels")

add_tensorflow_operation("store_tensor_to_file")
add_tensorflow_operation("py_hook")
