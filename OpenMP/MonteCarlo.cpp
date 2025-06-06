#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

int main() {
    const int num_points = 100000000;
    int count = 0;

    // Generate random points
    std::vector<float> random_x(num_points);
    std::vector<float> random_y(num_points);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < num_points; ++i) {
        random_x[i] = dist(rng);
        random_y[i] = dist(rng);
    }

    std::cout << "Running OpenMP Monte Carlo simulation to estimate Pi..." << std::endl;

    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Parallel computation using OpenMP
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < num_points; ++i) {
        float x = random_x[i];
        float y = random_y[i];
        if (x * x + y * y <= 1.0f) {
            count++;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "OpenMP Monte Carlo completed in " << duration.count() << " ms" << std::endl;

    // Calculate Pi
    float pi = 4.0f * count / num_points;
    std::cout << "Estimated value of Pi: " << pi << std::endl;

    return 0;
}