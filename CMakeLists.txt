cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(edge)

set(CPP_DIR "${PROJECT_SOURCE_DIR}/cpp")
set(CPP_SRC_DIR "${CPP_DIR}/src")
set(CPP_INCLUDE_DIR "${CPP_DIR}/include")
# *** Set CUDA path ***
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CONDA_PREFIX})
# *** Set LibTorch path ***
set(TORCHTRT_ROOT "${PROJECT_SOURCE_DIR}/third-party/TensorRT")
set(LIBTORCH_PATH "${PROJECT_SOURCE_DIR}/third-party/libtorch")
set(LIBTORCHTRT_PATH "${TORCHTRT_ROOT}/build/lib")

list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${TORCHTRT_ROOT}/build")
list(APPEND CMAKE_MODULE_PATH "${TORCHTRT_ROOT}/cmake/Modules")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

set(TORCHTRT_LIBRARIES "${LIBTORCHTRT_PATH}/libtorchtrt.so")
set(TORCHTRT_INCLUDE_DIRS "${TORCHTRT_ROOT}/cpp/include")
message(STATUS "TORCHTRT_LIBRARIES: ${TORCHTRT_LIBRARIES}")
set(TensorRT_LIBRARIES 
    "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.10"
    "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_plugin.so.10"
)
message(STATUS "TensorRT_LIBRARIES: ${TensorRT_LIBRARIES}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(benchmark "${CPP_SRC_DIR}/benchmark.cpp")

link_directories(PRIVATE ${LIBTORCHTRT_PATH}/lib)
target_link_libraries(benchmark "${TORCH_LIBRARIES}")
target_link_libraries(benchmark "${TensorRT_LIBRARIES}")
target_link_libraries(benchmark "${TORCHTRT_LIBRARIES}")
set_property(TARGET benchmark PROPERTY CXX_STANDARD 17)
include_directories(PRIVATE ${TORCHTRT_INCLUDE_DIRS})
include_directories(PRIVATE ${CPP_INCLUDE_DIR})
include_directories(PRIVATE ${TORCH_INCLUDE_DIRS})