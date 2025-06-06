#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <chrono>
#include "../Matrix.h"
#include <omp.h>

int main() {
    srand(static_cast<unsigned int>(time(0)));
    const int N = 1500;
    Matrix A = Matrix<float>::generateMatrixSystemEquations(N);
    std::vector<float> b(N);
    for (int i = 0; i < N; ++i) {
        b[i] = static_cast<float>(rand() % 1000000 + 1);
    }
    std::vector<float> x(N, 0);

    // Initialize r and p
    std::vector<float> r(N, 0.0f);
    std::vector<float> p(N, 0.0f);
    std::vector<float> Ap(N, 0.0f);

    // Compute initial r and p values
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A(i, j) * x[j];
        }
        p[i] = r[i] = b[i] - sum;
    }

    double rsold = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    std::cout << "Starting Conjugate Gradient method..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Main loop
    for (int iter = 0; iter < N; ++iter) {
        // Compute Ap = A * p
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += A(i, j) * p[j];
            }
            Ap[i] = sum;
        }

        // Compute alpha
        double denom = std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);
        if (std::fabs(denom) < std::numeric_limits<double>::epsilon()) {
            std::cerr << "Denominator too small, stopping iteration." << std::endl;
            break;
        }
        double alpha = rsold / denom;

        // Update x and r
        #pragma omp parallel for
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
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    std::cout << "Conjugate Gradient completed in " << elapsed.count() << " seconds." << std::endl;

    // std::cout << "Solution: ";
    // for (float xi : x) {
    //     std::cout << xi << " ";
    // }
    // std::cout << std::endl;

    return 0;
}