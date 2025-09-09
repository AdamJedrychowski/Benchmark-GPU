#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "../Matrix.h"

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

        // Perform reduction in local memory
        for (int offset = groupSize / 2; offset > 0; offset /= 2) {
            if (localId < offset) {
                localData[localId] += localData[localId + offset];
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
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Device device = devices.front();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, kernelSource);
        program.build({device});
        cl::Kernel mapKernel(program, "map");
        cl::Kernel reduceKernel(program, "reduce");

        std::vector<int> sizes = {33554432, 67108864, 134217728, 268435456, 536870912, 1073741824};

        for(int &n : sizes) {
            std::vector<int> input(n);
            int workGroupSize = 256, numWorkGroupSize = n / workGroupSize;
            
            for(auto& val : input) {
                val = rand() % 100;
            }

            cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, input.data());
            cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * n, nullptr);
            cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * numWorkGroupSize, nullptr);

            mapKernel.setArg(0, inputBuffer);
            mapKernel.setArg(1, outputBuffer);

            reduceKernel.setArg(0, outputBuffer);
            reduceKernel.setArg(1, cl::Local(sizeof(int) * workGroupSize));
            reduceKernel.setArg(2, resultBuffer);

            std::cout << "Running OpenCL MapReduce style data processing..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();

            cl::NDRange global(n);
            cl::NDRange local(256);
            queue.enqueueNDRangeKernel(mapKernel, cl::NullRange, global, local);
            queue.enqueueNDRangeKernel(reduceKernel, cl::NullRange, global, local);

            std::vector<int> results(numWorkGroupSize);
            queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(int) * numWorkGroupSize, results.data());

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;
            std::cout << "OpenCL MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;
            saveDurationToFile("results/MapReduce/OpenCL.txt", n, duration.count());

            int64_t result = std::accumulate(results.begin(), results.end(), (int64_t)0);
            std::cout << "Reduction result: " << result << std::endl;
        }
    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}