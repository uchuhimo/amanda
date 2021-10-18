cmake_minimum_required(VERSION 3.17...3.21)
project(amanda_tf_pybind)

include(CMakePrintHelpers)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
    cmake_print_variables(CMAKE_CXX_COMPILER_LAUNCHER)
endif()

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/modules)
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(TensorFlow REQUIRED)
find_package(pybind11 CONFIG)

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

include_directories(SYSTEM ${Python_INCLUDE_DIRS})
include_directories(SYSTEM ${Python_NumPy_INCLUDE_DIRS})
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})

pybind11_add_module(amanda_tf_pybind amanda_tf_pybind.cpp)
target_link_libraries(amanda_tf_pybind LINK_PUBLIC ${TensorFlow_LIBRARY})

add_tensorflow_operation("store_tensor_to_file")
# add_tensorflow_operation("py_hook")
