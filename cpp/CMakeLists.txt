cmake_minimum_required(VERSION 3.20)
project(parakeet_trt LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FindPackageHandleStandardArgs)

# TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h PATHS /usr/local/cuda/include /usr/include/x86_64-linux-gnu)
find_library(TRT_INFER_LIB nvinfer PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
find_library(TRT_ONNX_LIB nvonnxparser PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

# SentencePiece
find_package(PkgConfig REQUIRED)
pkg_check_modules(SP sentencepiece)

add_library(parakeet_trt SHARED
    src/parakeet_trt.cpp
    src/engine_manager.cpp
    src/tokenizer.cpp
    src/decoder.cpp
)

target_include_directories(parakeet_trt PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${TRT_INCLUDE_DIR}
    ${SP_INCLUDE_DIRS}
)

target_link_libraries(parakeet_trt PRIVATE 
    ${TRT_INFER_LIB} 
    ${TRT_ONNX_LIB}
    ${SP_LIBRARIES}
)

# Install rules if needed
