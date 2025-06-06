#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <chrono>
#include "../Matrix.h"

// CUDA kernel for Conjugate Gradient
__global__ void conjugateGradientKernel(
    const float* A, const float* p, float* Ap, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[gid * N + j] * p[j];
        }
        Ap[gid] = sum;
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    const int N = 1500;
    Matrix<float> A = Matrix<float>::generateMatrixSystemEquations(N);
    std::vector<float> b(N);
    for (int i = 0; i < N; ++i) {
        b[i] = static_cast<float>(rand() % 1000000 + 1);
    }
    std::vector<float> x(N, 0);

    try {
        // Allocate device memory
        float *d_A, *d_b, *d_x, *d_p, *d_Ap;
        cudaMalloc(&d_A, sizeof(float) * N * N);
        cudaMalloc(&d_b, sizeof(float) * N);
        cudaMalloc(&d_x, sizeof(float) * N);
        cudaMalloc(&d_p, sizeof(float) * N);
        cudaMalloc(&d_Ap, sizeof(float) * N);

        // Copy data to device
        cudaMemcpy(d_A, A.data(), sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

        // Initialize r and p
        std::vector<float> r(N, 0.0f);
        std::vector<float> p(N, 0.0f);
        std::vector<float> Ap(N, 0.0f);

        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += A(i, j) * x[j];
            }
            p[i] = r[i] = b[i] - sum;
        }

        cudaMemcpy(d_p, p.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

        double rsold = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

        std::cout << "Starting Conjugate Gradient method..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        // Main loop
        for (int iter = 0; iter < N; ++iter) {
            // Launch kernel
            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            conjugateGradientKernel<<<gridSize, blockSize>>>(d_A, d_p, d_Ap, N);
            cudaDeviceSynchronize();

            // Read back Ap
            cudaMemcpy(Ap.data(), d_Ap, sizeof(float) * N, cudaMemcpyDeviceToHost);

            // Compute alpha
            double denom = std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);
            if (std::fabs(denom) < std::numeric_limits<double>::epsilon()) {
                std::cerr << "Denominator too small, stopping iteration." << std::endl;
                break;
            }
            double alpha = rsold / denom;

            // Update x and r
            for (int i = 0; i < N; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double rsnew = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

            if (std::sqrt(rsnew) < 1e-6) {
                std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
                break;
            }

            // Update p
            double beta = rsnew / rsold;
            for (int i = 0; i < N; ++i) {
                p[i] = r[i] + beta * p[i];
            }

            cudaMemcpy(d_p, p.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

            rsold = rsnew;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Conjugate Gradient completed in " << elapsed.count() << " seconds." << std::endl;

        // Copy result back to host
        cudaMemcpy(x.data(), d_x, sizeof(float) * N, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_p);
        cudaFree(d_Ap);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}