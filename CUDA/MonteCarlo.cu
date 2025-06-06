#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// CUDA kernel for Monte Carlo simulation
__global__ void monte_carlo_pi(const float* random_x, const float* random_y, int* count, int num_points) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_points) {
        float x = random_x[id];
        float y = random_y[id];
        if (x * x + y * y <= 1.0f) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    const int num_points = 100000000;
    std::vector<float> random_x(num_points);
    std::vector<float> random_y(num_points);
    int count = 0;

    // Generate random points
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < num_points; ++i) {
        random_x[i] = dist(rng);
        random_y[i] = dist(rng);
    }

    // Allocate device memory
    float *d_random_x, *d_random_y;
    int *d_count;
    cudaMalloc(&d_random_x, sizeof(float) * num_points);
    cudaMalloc(&d_random_y, sizeof(float) * num_points);
    cudaMalloc(&d_count, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_random_x, random_x.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_random_y, random_y.data(), sizeof(float) * num_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Running CUDA Monte Carlo simulation to estimate Pi..." << std::endl;

    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    monte_carlo_pi<<<blocksPerGrid, threadsPerBlock>>>(d_random_x, d_random_y, d_count, num_points);

    // Copy result back to host
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "CUDA Monte Carlo completed in " << duration.count() << " ms" << std::endl;

    // Calculate Pi
    float pi = 4.0f * count / num_points;
    std::cout << "Estimated value of Pi: " << pi << std::endl;

    // Free device memory
    cudaFree(d_random_x);
    cudaFree(d_random_y);
    cudaFree(d_count);

    return 0;
}