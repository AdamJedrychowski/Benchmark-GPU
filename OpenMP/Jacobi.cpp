#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "../Matrix.h"
#include <omp.h>
#include <chrono>

int main() {
    srand(1000);

    int maxIterations = 1000;
    float tolerance = 1e-6;
    std::vector<int> sizes = {1250, 2500, 5000, 10000, 20000, 40000};

    for(int &n : sizes) {
        Matrix<float> A(n);
        Matrix<float>::generateMatrixSystemEquations(A);
        std::vector<float> b(n);
        for (int i = 0; i < n; ++i) {
            b[i] = static_cast<float>(rand() % 1000000 + 1);
        }
        std::vector<float> x(n, 0.0f), x_new(n, 0.0f);

        std::cout << "Starting Jacobi Iteration with OpenMP..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < maxIterations; ++iter) {
            float error = 0.0f;

            #pragma omp parallel for reduction(+:error)
            for (int i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        sum += A(i, j) * x[j];
                    }
                }
                x_new[i] = (b[i] - sum) / A(i, i);
                error += std::abs(x_new[i] - x[i]);
            }

            if (error < tolerance) {
                break;
            }

            for (int i = 0; i < n; ++i) {
                x[i] = x_new[i];
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "Jacobi Iteration completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/Jacobi/OpenMP.txt", n, duration.count());
    }

    return 0;
}