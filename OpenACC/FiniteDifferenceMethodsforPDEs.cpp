#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <openacc.h>
#include "../Matrix.h"

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

        std::cout << "Running OpenACC heat equation solver..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        double* u_ptr = u.data();
        double* u_next_ptr = u_next.data();

        #pragma acc data copy(u_ptr[0:Nx]) create(u_next_ptr[0:Nx])
        {
            for (int n = 0; n < Nt; ++n) {
                #pragma acc parallel loop present(u_ptr, u_next_ptr)
                for (int i = 1; i < Nx - 1; ++i) {
                    u_next_ptr[i] = u_ptr[i] + r * (u_ptr[i + 1] - 2 * u_ptr[i] + u_ptr[i - 1]);
                }

                // Set boundary conditions
                #pragma acc parallel loop present(u_next_ptr)
                for (int i = 0; i < Nx; ++i) {
                    if (i == 0 || i == Nx - 1) {
                        u_next_ptr[i] = 0.0;
                    }
                }

                // Swap arrays
                #pragma acc parallel loop present(u_ptr, u_next_ptr)
                for (int i = 0; i < Nx; ++i) {
                    u_ptr[i] = u_next_ptr[i];
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "OpenACC heat equation solver completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/FiniteDifferenceMethodsforPDEs/OpenACC.txt", Nt, duration.count());
    }

    return 0;
}