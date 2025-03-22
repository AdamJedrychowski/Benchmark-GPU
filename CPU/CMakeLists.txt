set(CPU_SOURCES

)

set(CPU_HEADERS
    
)

add_executable(cpu_benchmark cpu.cpp)

set_target_properties(cpu_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)