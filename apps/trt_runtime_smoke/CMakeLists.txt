cmake_minimum_required(VERSION 3.20)
project(trt_runtime_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PARAKEET_MOCK "Build without CUDA/TensorRT (smoke app becomes a no-op)" OFF)

add_executable(trt_runtime_smoke
  main.cpp
)

target_compile_definitions(trt_runtime_smoke PRIVATE $<$<BOOL:${PARAKEET_MOCK}>:PARAKEET_MOCK>)

if(PARAKEET_MOCK)
  message(STATUS "Building trt_runtime_smoke in MOCK mode (no CUDA/TRT required).")
  target_compile_options(trt_runtime_smoke PRIVATE -Wall -Wextra -Wpedantic)
  return()
endif()

# TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h
  PATHS
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/local/include
    /opt/nvidia/tensorrt/include
)
find_library(TRT_INFER_LIB nvinfer
  PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/cuda/lib64
    /opt/nvidia/tensorrt/lib
)

# CUDA runtime (cudart)
find_package(CUDAToolkit QUIET)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_INFER_LIB)
  message(FATAL_ERROR "TensorRT not found. Install TensorRT dev packages or configure with -DPARAKEET_MOCK=ON.")
endif()
if(NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "CUDA Toolkit not found (needed for cuda runtime). Install CUDA or configure with -DPARAKEET_MOCK=ON.")
endif()

target_include_directories(trt_runtime_smoke PRIVATE ${TRT_INCLUDE_DIR})
target_link_libraries(trt_runtime_smoke PRIVATE ${TRT_INFER_LIB} CUDA::cudart)
target_compile_options(trt_runtime_smoke PRIVATE -Wall -Wextra -Wpedantic)


