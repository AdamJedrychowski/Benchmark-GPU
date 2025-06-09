#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "../Matrix.h"

__global__ void mapKernel(const int* input, int* output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        output[id] = input[id] * input[id];
    }
}

__global__ void reduceKernel(int* data, int* result, int size) {
    extern __shared__ int sharedData[];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;

    if (id < size) {
        sharedData[localId] = data[id];
    } else {
        sharedData[localId] = 0;
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (localId < offset) {
            sharedData[localId] += sharedData[localId + offset];
        }
        __syncthreads();
    }

    if (localId == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

int main() {
    std::vector<int> sizes = {33554432, 67108864, 134217728, 268435456, 536870912, 1073741824};

    for(int &n : sizes) {
        const int workGroupSize = 256;
        const int numWorkGroups = (n + workGroupSize - 1) / workGroupSize;

        std::vector<int> input(n);
        for (auto& val : input) {
            val = rand() % 100;
        }

        std::vector<int> results(numWorkGroups);

        int *d_input, *d_output, *d_result;

        cudaMalloc(&d_input, sizeof(int) * n);
        cudaMalloc(&d_output, sizeof(int) * n);
        cudaMalloc(&d_result, sizeof(int) * numWorkGroups);

        cudaMemcpy(d_input, input.data(), sizeof(int) * n, cudaMemcpyHostToDevice);

        std::cout << "Running CUDA MapReduce style data processing..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        mapKernel<<<(n + workGroupSize - 1) / workGroupSize, workGroupSize>>>(d_input, d_output, n);
        cudaDeviceSynchronize();

        reduceKernel<<<numWorkGroups, workGroupSize, sizeof(int) * workGroupSize>>>(d_output, d_result, n);
        cudaDeviceSynchronize();

        cudaMemcpy(results.data(), d_result, sizeof(int) * numWorkGroups, cudaMemcpyDeviceToHost);

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "CUDA MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/MapReduce/CUDA.txt", n, duration.count());

        int64_t result = std::accumulate(results.begin(), results.end(), (int64_t)0);
        std::cout << "Reduction result: " << result << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_result);
    }

    return 0;
}