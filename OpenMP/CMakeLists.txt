find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(openmp_benchmark openmp.cpp)

set_target_properties(openmp_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

function(add_openmp_benchmark name)
    add_executable(${name}MP ${name}.cpp ../Matrix.h)
    target_link_libraries(${name}MP PUBLIC OpenCL-sdk-lib)
    set_target_properties(${name}MP
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_openmp_benchmark(FiniteDifferenceMethodsforPDEs)
add_openmp_benchmark(Jacobi)
add_openmp_benchmark(MapReduce)
add_openmp_benchmark(MonteCarlo)
add_openmp_benchmark(PageRank)