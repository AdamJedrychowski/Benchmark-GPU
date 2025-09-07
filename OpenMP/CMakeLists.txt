find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

function(add_openmp_benchmark name)
    add_executable(${name}MP ${name}.cpp ../Matrix.h)
    target_link_libraries(${name}MP PUBLIC OpenMP::OpenMP_CXX)
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