include(ExternalProject)

if (WIN32)
    ExternalProject_Add(OpenCL-build
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
        GIT_TAG v2023.04.17
        GIT_SHALLOW ON
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -D BUILD_TESTING=OFF -D BUILD_DOCS=OFF -D OPENCL_SDK_BUILD_SAMPLES=OFF
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
else()
    ExternalProject_Add(OpenCL-build
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
        GIT_TAG v2023.04.17
        GIT_SHALLOW ON
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -D BUILD_TESTING=OFF -D BUILD_DOCS=OFF -D OPENCL_SDK_BUILD_SAMPLES=OFF
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
    )
endif()
ExternalProject_Get_Property(OpenCL-build INSTALL_DIR)

set(opencl_install_dir ${INSTALL_DIR})

if (WIN32)
    add_library(OpenCL-bin STATIC IMPORTED)
    set_target_properties (OpenCL-bin
        PROPERTIES
        IMPORTED_LOCATION ${opencl_install_dir}/lib/OpenCL.lib
    )
else ()
    add_library(OpenCL-bin SHARED IMPORTED)
    set_target_properties (OpenCL-bin
        PROPERTIES
        IMPORTED_LOCATION ${opencl_install_dir}/lib/libOpenCL.so.1.2
    )
endif ()
add_dependencies(OpenCL-bin PUBLIC OpenCL-build)

add_library(OpenCL-sdk-lib INTERFACE)
target_include_directories(OpenCL-sdk-lib INTERFACE ${opencl_install_dir}/include)
target_link_libraries(OpenCL-sdk-lib INTERFACE OpenCL-bin)

add_definitions(-DCL_HPP_ENABLE_EXCEPTIONS)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=300)

function(add_opencl_benchmark name)
    add_executable(${name}CL ${name}.cpp ../Matrix.h)
    target_link_libraries(${name}CL PUBLIC OpenCL-sdk-lib)
    set_target_properties(${name}CL
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_opencl_benchmark(PageRank)
add_opencl_benchmark(FiniteDifferenceMethodsforPDEs)
add_opencl_benchmark(Jacobi)
add_opencl_benchmark(MapReduce)
add_opencl_benchmark(MonteCarlo)

add_opencl_benchmark(DeviceInfo)