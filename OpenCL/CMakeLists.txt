include(ExternalProject)

ExternalProject_Add(OpenCL-build
GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
    GIT_TAG v2023.04.17
    GIT_SHALLOW ON
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -D BUILD_TESTING=OFF -D BUILD_DOCS=OFF -D OPENCL_SDK_BUILD_SAMPLES=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
)
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


set(Opencl_SOURCES
)

set(Opencl_HEADERS
)

# add_library(Opencl-lib STATIC ${Opencl_SOURCES} ${Opencl_HEADERS})
# target_link_libraries(Opencl-lib PUBLIC OpenCL-sdk-lib)

add_executable(opencl_benchmark opencl.cpp)
target_link_libraries(opencl_benchmark PUBLIC OpenCL-sdk-lib)

set_target_properties(opencl_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)