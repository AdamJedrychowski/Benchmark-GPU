# It is possibleto run it using NVIDIA HPC SDK
# Download docker pull nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu22.04
# Run docker run --rm -it --gpus all -w /work nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu22.04 bash

find_package(OpenACC REQUIRED)

function(add_openacc_benchmark name)
    add_executable(${name}ACC ${name}.cpp ../Matrix.h)
    
    # Set OpenACC compiler flags
    if(OpenACC_CXX_FOUND)
        target_compile_options(${name}ACC PRIVATE ${OpenACC_CXX_FLAGS})
        target_link_libraries(${name}ACC PRIVATE ${OpenACC_CXX_FLAGS})
    else()
        # Fallback for NVIDIA HPC SDK or PGI compilers
        if(CMAKE_CXX_COMPILER_ID MATCHES "PGI|NVHPC")
            target_compile_options(${name}ACC PRIVATE -acc)
            target_link_options(${name}ACC PRIVATE -acc)
        endif()
    endif()
    
    set_target_properties(${name}ACC
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_openacc_benchmark(PageRank)
add_openacc_benchmark(FiniteDifferenceMethodsforPDEs)
add_openacc_benchmark(Jacobi)
add_openacc_benchmark(MapReduce)
add_openacc_benchmark(MonteCarlo)