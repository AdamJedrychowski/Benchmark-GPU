set(CMAKE_CUDA_STANDARD 11)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_benchmark cuda.cu)
target_link_libraries(cuda_benchmark ${CUDA_LIBRARIES})

set_target_properties(cuda_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

function(add_cuda_benchmark name)
    add_executable(${name}CUDA ${name}.cu ../Matrix.h)
    target_link_libraries(${name}CUDA PUBLIC ${CUDA_LIBRARIES})
    set_target_properties(${name}CUDA
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_cuda_benchmark(FiniteDifferenceMethodsforPDEs)
add_cuda_benchmark(Jacobi)
add_cuda_benchmark(MapReduce)
add_cuda_benchmark(MonteCarlo)
add_cuda_benchmark(PageRank)