set(CMAKE_CUDA_STANDARD 11)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_benchmark cuda.cu)
target_link_libraries(cuda_benchmark ${CUDA_LIBRARIES})

set_target_properties(cuda_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)