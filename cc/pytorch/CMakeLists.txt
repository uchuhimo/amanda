cmake_minimum_required(VERSION 3.17...3.21)
project(amanda_torch_pybind)

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

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(SYSTEM ${Python_INCLUDE_DIRS})

pybind11_add_module(amanda_torch_pybind amanda_torch_pybind.cpp)
target_link_libraries(amanda_torch_pybind PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(amanda_torch_pybind PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(amanda_torch_pybind PRIVATE ${TORCH_PYTHON_LIBRARY})
