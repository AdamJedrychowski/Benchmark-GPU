#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include "../Matrix.h"

int main() {
    std::vector<int> num_points_list = {62500000, 125000000, 250000000, 500000000, 1000000000, 2000000000}; 
    
    for(int &num_points : num_points_list) {
        int count = 0;
        std::vector<float> random_x(num_points);
        std::vector<float> random_y(num_points);

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < num_points; ++i) {
            random_x[i] = dist(rng);
            random_y[i] = dist(rng);
        }

        std::cout << "Running OpenMP Monte Carlo simulation to estimate Pi..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:count)
        for (int i = 0; i < num_points; ++i) {
            float x = random_x[i];
            float y = random_y[i];
            if (x * x + y * y <= 1.0f) {
                count++;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "OpenMP Monte Carlo completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/MonteCarlo/OpenMP.txt", num_points, duration.count());

        float pi = 4.0f * count / num_points;
        std::cout << "Estimated value of Pi: " << pi << std::endl;
    }

    return 0;
}