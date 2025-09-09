#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "../Matrix.h"
#include <chrono>

int main() {
    srand(1000);

    int maxIterations = 1000;
    float tolerance = 1e-6;
    std::vector<int> sizes = {1250, 2500, 5000, 10000, 20000}; // , 40000

    for(int &n : sizes) {
        Matrix<float> A(n);
        Matrix<float>::generateMatrixSystemEquations(A);
        std::vector<float> b(n);
        for (int i = 0; i < n; ++i) {
            b[i] = static_cast<float>(rand() % 1000000 + 1);
        }
        std::vector<float> x(n, 0.0f), x_new(n, 0.0f);

        std::cout << "Starting Jacobi Iteration with OpenACC..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        // Get pointers to data for OpenACC
        float* A_data = A.data();
        float* x_data = x.data();
        float* x_new_data = x_new.data();
        float* b_data = b.data();

        // Copy data to GPU
        #pragma acc data copyin(A_data[0:n*n], b_data[0:n]) copy(x_data[0:n], x_new_data[0:n])
        {
            for (int iter = 0; iter < maxIterations; ++iter) {
                float error = 0.0f;

                #pragma acc parallel loop reduction(+:error)
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    #pragma acc loop seq
                    for (int j = 0; j < n; ++j) {
                        if (i != j) {
                            sum += A_data[i * n + j] * x_data[j];
                        }
                    }
                    x_new_data[i] = (b_data[i] - sum) / A_data[i * n + i];
                    error += std::abs(x_new_data[i] - x_data[i]);
                }

                if (error < tolerance) {
                    break;
                }

                #pragma acc parallel loop
                for (int i = 0; i < n; ++i) {
                    x_data[i] = x_new_data[i];
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "Jacobi Iteration completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/Jacobi/OpenACC.txt", n, duration.count());
    }

    return 0;
}