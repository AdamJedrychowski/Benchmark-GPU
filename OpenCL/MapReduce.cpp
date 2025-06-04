#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

// Kernel source
const char* kernelSource = R"(
    __kernel void map(__global const int* input, __global int* output) {
        int id = get_global_id(0);
        output[id] = input[id] * input[id];
    }

    __kernel void reduce(__global int* data, __local int* localData, __global int* result) {
        int id = get_global_id(0);
        int localId = get_local_id(0);
        int groupSize = get_local_size(0);

        // Load data into local memory
        localData[localId] = data[id];
        barrier(CLK_LOCAL_MEM_FENCE);

        int prevOffset = groupSize;
        // Perform reduction in local memory
        for (int offset = groupSize / 2; offset > 0; offset /= 2) {
            if (localId < offset) {
                if(localId == offset - 1 && offset*2 != prevOffset) {
                    localData[localId] += localData[localId + offset + 1];
                }
                localData[localId] += localData[localId + offset];
                prevOffset = offset;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Write result of reduction for this group
        if (localId == 0) {
            result[get_group_id(0)] = localData[0];
        }
    }
    )";

int main() {
    // Input data
    std::vector<int> input(131072);
    int workGroupSize = 256, numWorkGroupSize = input.size() / workGroupSize;
    for(auto& val : input) {
        val = rand() % 100;
    }

    try {
        // Get all platforms (drivers)
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();

        // Get all devices for the platform
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Device device = devices.front();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Build the program
        cl::Program program(context, kernelSource);
        program.build({device});

        // Buffers
        cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * input.size(), input.data());
        cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * input.size(), nullptr);
        cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * numWorkGroupSize, nullptr);

        // Map kernel
        cl::Kernel mapKernel(program, "map");
        mapKernel.setArg(0, inputBuffer);
        mapKernel.setArg(1, outputBuffer);

        // Reduce kernel
        cl::Kernel reduceKernel(program, "reduce");
        reduceKernel.setArg(0, outputBuffer);
        reduceKernel.setArg(1, cl::Local(sizeof(int) * workGroupSize)); // Local memory size
        reduceKernel.setArg(2, resultBuffer);

        std::cout << "Running OpenCL MapReduce style data processing..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        // Execute map kernel
        cl::NDRange global(input.size());
        cl::NDRange local(256);
        queue.enqueueNDRangeKernel(mapKernel, cl::NullRange, global, local);

        // Execute reduce kernel
        queue.enqueueNDRangeKernel(reduceKernel, cl::NullRange, global, local);

        std::vector<int> results(numWorkGroupSize);
        queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(int) * numWorkGroupSize, results.data());

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "OpenCL MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;

        int64_t result = std::accumulate(results.begin(), results.end(), (int64_t)0);
        std::cout << "Reduction result: " << result << std::endl;

    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}