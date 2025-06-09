#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <chrono>
#include <cuda_runtime.h>
#include "../Matrix.h"

__global__ void heatEquationKernel(double* u, double* u_next, double r, int Nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < Nx - 1) {
        u_next[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1]);
    }
}

int main() {
    const double alpha = 0.01; // Heat conduction coefficient
    const double L = 1.0;        // Length of the rod
    const double T = 1.0;        // Total simulation time
    const int Nx = 20;           // Grid sizes
    const double dx = L / (Nx - 1);

    // Number of time steps
    std::vector<int> Nt_values = {50000, 100000, 200000, 400000, 800000, 1600000};
        
    for(int & Nt : Nt_values) {
        const double dt = T / Nt;
        const double r = alpha * dt / (dx * dx);  // Stability condition
        std::vector<double> u(Nx);
        std::vector<double> u_next(Nx);

        for (int i = 0; i < Nx; ++i) {
            double x = i * dx;
            u[i] = sin(M_PI * x);
        }

        double *d_u, *d_u_next;
        cudaMalloc(&d_u, Nx * sizeof(double));
        cudaMalloc(&d_u_next, Nx * sizeof(double));

        cudaMemcpy(d_u, u.data(), Nx * sizeof(double), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (Nx + blockSize - 1) / blockSize;

        std::cout << "Running CUDA heat equation solver..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int n = 0; n < Nt; ++n) {
            heatEquationKernel<<<numBlocks, blockSize>>>(d_u, d_u_next, r, Nx);
            cudaMemcpy(d_u, d_u_next, Nx * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(u.data(), d_u, Nx * sizeof(double), cudaMemcpyDeviceToHost);

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "CUDA heat equation solver completed in " << duration.count() << " ms" << std::endl;
        saveDurationToFile("results/FiniteDifferenceMethodsforPDEs/CUDA.txt", Nt, duration.count());

        cudaFree(d_u);
        cudaFree(d_u_next);
    }

    return 0;
}