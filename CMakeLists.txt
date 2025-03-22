cmake_minimum_required(VERSION 3.22)
project(Benchmark-GPU-FrameWorks LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_subdirectory(CPU)
add_subdirectory(CUDA)
add_subdirectory(OpenCL)
add_subdirectory(OpenMP)
add_subdirectory(Sycl)