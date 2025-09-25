#define _USE_MATH_DEFINES
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "../Matrix.h"

int main() {
    const double alpha = 0.01; // Heat conduction coefficient
    const double L = 1.0;        // Length of the rod
    const double T = 1.0;        // Total simulation time
    const int Nx = 20;           // Grid sizes
    const double dx = L / (Nx - 1);

    try {
        sycl::queue queue{sycl::gpu_selector{}};
        
        std::cout << "Running on device: " 
                  << queue.get_device().get_info<sycl::info::device::name>() 
                  << std::endl;

        // Number of time steps
        std::vector<int> Nt_values = {50000, 100000, 200000, 400000, 800000, 1600000};
        
        for(int Nt : Nt_values) {
            const double dt = T / Nt;
            const double r = alpha * dt / (dx * dx);  // Stability condition
            
            std::cout << "Running SYCL heat equation solver for Nt = " << Nt << "..." << std::endl;
            
            std::vector<double> u_host(Nx);
            for (int i = 0; i < Nx; ++i) {
                double x = i * dx;
                u_host[i] = sin(M_PI * x);
            }

            double* u = sycl::malloc_device<double>(Nx, queue);
            double* u_next = sycl::malloc_device<double>(Nx, queue);

            queue.memcpy(u, u_host.data(), sizeof(double) * Nx).wait();

            auto startTime = std::chrono::high_resolution_clock::now();

            for (int n = 0; n < Nt; ++n) {
                queue.parallel_for(sycl::range<1>(Nx), [=](sycl::id<1> i) {
                    int idx = i[0];
                    if (idx > 0 && idx < Nx - 1) {
                        u_next[idx] = u[idx] + r * (u[idx+1] - 2*u[idx] + u[idx-1]);
                    }
                }).wait();

                // Swap buffers by copying u_next to u
                queue.memcpy(u, u_next, sizeof(double) * Nx).wait();
            }
            queue.memcpy(u_host.data(), u, sizeof(double) * Nx).wait();

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;
            std::cout << "SYCL heat equation solver completed in " << duration.count() << " seconds." << std::endl;
            saveDurationToFile("results/FiniteDifferenceMethodsforPDEs/Sycl.txt", Nt, duration.count());

            sycl::free(u, queue);
            sycl::free(u_next, queue);
        }
    } catch (sycl::exception& e) {
        std::cout << "SYCL error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}