# Find oneAPI installation
if(NOT DEFINED ONEAPI_ROOT)
    if(DEFINED ENV{ONEAPI_ROOT})
        set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})
    else()
        if(WIN32)
            set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
        else()
            set(ONEAPI_ROOT "/opt/intel/oneapi")
        endif()
    endif()
endif()

# Configure SYCL compiler
if(WIN32)
    add_custom_target(sycl_benchmark ALL
        COMMAND cmd /c "icpx -fsycl -std=c++17 -L \"C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib\" -o ${CMAKE_BINARY_DIR}/bin/Debug/sycl_direct.exe sycl.cpp"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building SYCL with Intel oneAPI"
        SOURCES sycl.cpp
        BYPRODUCTS "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/sycl_direct.exe"
    )
else()
    set(CMAKE_CXX_COMPILER "${ONEAPI_ROOT}/compiler/latest/linux/bin/icpx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -std=c++17")
  
    include_directories(
        "${ONEAPI_ROOT}/compiler/latest/include"
        "${ONEAPI_ROOT}/compiler/latest/include/sycl"
    )

    add_executable(sycl_benchmark sycl.cpp)

    set_target_properties(sycl_benchmark
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()