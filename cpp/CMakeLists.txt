cmake_minimum_required(VERSION 3.20)
project(parakeet_trt LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FindPackageHandleStandardArgs)

# Options
option(PARAKEET_MOCK "Build with mock implementation for testing without CUDA/TRT" OFF)

if(PARAKEET_MOCK)
    message(STATUS "Building in MOCK mode")
    add_definitions(-DPARAKEET_MOCK)
    add_library(parakeet_trt SHARED
        src/mock_lib.cpp
        src/trt_asr.cpp
    )
else()
    # TensorRT
    find_path(TRT_INCLUDE_DIR NvInfer.h PATHS /usr/local/cuda/include /usr/include/x86_64-linux-gnu /opt/nvidia/tensorrt/include)
    find_library(TRT_INFER_LIB nvinfer PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /opt/nvidia/tensorrt/lib)
    find_library(TRT_ONNX_LIB nvonnxparser PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /opt/nvidia/tensorrt/lib)
    find_library(CUDART_LIB cudart PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

    if(NOT TRT_INCLUDE_DIR OR NOT TRT_INFER_LIB)
        message(FATAL_ERROR "TensorRT not found. Deeply sorry. Please install it or run with -DPARAKEET_MOCK=ON")
    endif()

    add_library(parakeet_trt SHARED
        src/parakeet_trt.cpp
        src/tokenizer.cpp
        src/decoder.cpp
        src/trt_asr.cpp
    )
endif()

target_include_directories(parakeet_trt PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${TRT_INCLUDE_DIR}
)

    target_link_libraries(parakeet_trt PRIVATE 
        ${TRT_INFER_LIB} 
        ${TRT_ONNX_LIB}
        ${CUDART_LIB}
    )

# Small deterministic smoke test for the tensor-only greedy decode loop.
add_executable(greedy_decode_smoke
    src/greedy_decode_smoke.cpp
)
target_include_directories(greedy_decode_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(greedy_decode_smoke PRIVATE -Wall -Wextra -Wpedantic)

# Install rules if needed
