#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "../Matrix.h"
#include <omp.h>
#include <chrono>

int main() {
    srand(static_cast<unsigned int>(time(0)));
    int n = 150;
    Matrix A = Matrix<float>::generateMatrixSystemEquations(n);
    std::vector<float> b(n);
    for (int i = 0; i < n; ++i) {
        b[i] = static_cast<float>(rand() % 1000000 + 1);
    }
    std::vector<float> x(n, 0.0f), x_new(n, 0.0f);

    int maxIterations = 1000;
    float tolerance = 1e-6;

    std::cout << "Starting Jacobi Iteration with OpenMP..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < maxIterations; ++iter) {
        float error = 0.0f;

        // Parallelize the Jacobi iteration using OpenMP
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

        // Update x with x_new
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] = x_new[i];
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "Jacobi Iteration completed in " << duration.count() << " seconds." << std::endl;
    std::cout << "Solution: ";
    for (int i = 0; i < n; ++i) {
        std::cout << x_new[i] << " ";
    }

    return 0;
}