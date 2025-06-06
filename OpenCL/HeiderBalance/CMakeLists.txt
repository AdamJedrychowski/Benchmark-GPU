set(Balancer_SOURCES
    main.cpp
)

set(Balancer_HEADERS
    ../../Matrix.h
)



add_library(Balancer-lib STATIC ${Balancer_SOURCES} ${Balancer_HEADERS})
target_link_libraries(Balancer-lib PUBLIC OpenCL-sdk-lib)

add_executable(Balancer main.cpp)
target_link_libraries(Balancer PUBLIC Balancer-lib)

# install(TARGETS Balancer DESTINATION ${PROJECT_SOURCE_DIR}/../runtime)
# install(FILES OpenCLRun.cl DESTINATION ${PROJECT_SOURCE_DIR}/../runtime)

file(COPY OpenCLRun.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)