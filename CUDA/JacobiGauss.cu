#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include "../Matrix.h"

// CUDA kernel for Jacobi Iteration
__global__ void jacobiKernel(const float* A, const float* b, float* x, float* x_new, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                sum += A[i * n + j] * x[j];
            }
        }
        x_new[i] = (b[i] - sum) / A[i * n + i];
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));
    int n = 150;
    Matrix<float> A = Matrix<float>::generateMatrixSystemEquations(n);
    std::vector<float> b(n);
    for (int i = 0; i < n; ++i) {
        b[i] = static_cast<float>(rand() % 1000000 + 1);
    }
    std::vector<float> x(n, 0.0f), x_new(n, 0.0f);

    int maxIterations = 1000;
    float tolerance = 1e-6;

    float *d_A, *d_b, *d_x, *d_x_new;

    // Allocate device memory
    cudaMalloc(&d_A, sizeof(float) * n * n);
    cudaMalloc(&d_b, sizeof(float) * n);
    cudaMalloc(&d_x, sizeof(float) * n);
    cudaMalloc(&d_x_new, sizeof(float) * n);

    // Copy data to device
    cudaMemcpy(d_A, A.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    std::cout << "Starting Jacobi Iteration..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < maxIterations; ++iter) {
        jacobiKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_b, d_x, d_x_new, n);
        cudaDeviceSynchronize();

        cudaMemcpy(x_new.data(), d_x_new, sizeof(float) * n, cudaMemcpyDeviceToHost);

        float error = 0.0f;
        for (int i = 0; i < n; ++i) {
            error += std::abs(x_new[i] - x[i]);
        }
        if (error < tolerance) {
            break;
        }

        cudaMemcpy(d_x, d_x_new, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "Jacobi Iteration completed in " << duration.count() << " seconds." << std::endl;
    std::cout << "Solution: ";
    for (int i = 0; i < n; ++i) {
        std::cout << x_new[i] << " ";
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_x_new);

    return 0;
}