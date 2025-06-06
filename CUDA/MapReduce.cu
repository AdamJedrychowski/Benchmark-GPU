#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

// CUDA kernel for the map operation
__global__ void mapKernel(const int* input, int* output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        output[id] = input[id] * input[id];
    }
}

// CUDA kernel for the reduce operation
__global__ void reduceKernel(int* data, int* result, int size) {
    extern __shared__ int sharedData[];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;

    // Load data into shared memory
    if (id < size) {
        sharedData[localId] = data[id];
    } else {
        sharedData[localId] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (localId < offset) {
            sharedData[localId] += sharedData[localId + offset];
        }
        __syncthreads();
    }

    // Write result of reduction for this block
    if (localId == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

int main() {
    // Input data
    const int dataSize = 131072;
    const int workGroupSize = 256;
    const int numWorkGroups = (dataSize + workGroupSize - 1) / workGroupSize;

    std::vector<int> input(dataSize);
    for (auto& val : input) {
        val = rand() % 100;
    }

    std::vector<int> results(numWorkGroups);

    int *d_input, *d_output, *d_result;

    // Allocate device memory
    cudaMalloc(&d_input, sizeof(int) * dataSize);
    cudaMalloc(&d_output, sizeof(int) * dataSize);
    cudaMalloc(&d_result, sizeof(int) * numWorkGroups);

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), sizeof(int) * dataSize, cudaMemcpyHostToDevice);

    std::cout << "Running CUDA MapReduce style data processing..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Execute map kernel
    mapKernel<<<(dataSize + workGroupSize - 1) / workGroupSize, workGroupSize>>>(d_input, d_output, dataSize);
    cudaDeviceSynchronize();

    // Execute reduce kernel
    reduceKernel<<<numWorkGroups, workGroupSize, sizeof(int) * workGroupSize>>>(d_output, d_result, dataSize);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(results.data(), d_result, sizeof(int) * numWorkGroups, cudaMemcpyDeviceToHost);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "CUDA MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;

    int64_t result = std::accumulate(results.begin(), results.end(), (int64_t)0);
    std::cout << "Reduction result: " << result << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_result);

    return 0;
}