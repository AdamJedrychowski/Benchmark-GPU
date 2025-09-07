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

        #pragma acc data copy(u[0:Nx]) create(u_next[0:Nx])
        for (int n = 0; n < Nt; ++n) {
            #pragma acc parallel loop
            for (int i = 1; i < Nx - 1; ++i) {
                u_next[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1]);
            }

            #pragma acc parallel loop
            for (int i = 0; i < Nx; ++i) {
                u[i] = u_next[i];
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "OpenACC heat equation solver completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/FiniteDifferenceMethodsforPDEs/OpenACC.txt", Nt, duration.count());
    }

    return 0;
}