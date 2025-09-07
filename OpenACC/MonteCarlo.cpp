#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <openacc.h>
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

        std::cout << "Running OpenACC Monte Carlo simulation to estimate Pi..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        float* x_ptr = random_x.data();
        float* y_ptr = random_y.data();

        #pragma acc data copyin(x_ptr[0:num_points], y_ptr[0:num_points])
        {
            #pragma acc parallel loop reduction(+:count)
            for (int i = 0; i < num_points; ++i) {
                float x = x_ptr[i];
                float y = y_ptr[i];
                if (x * x + y * y <= 1.0f) {
                    count++;
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "OpenACC Monte Carlo completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/MonteCarlo/OpenACC.txt", num_points, duration.count());

        float pi = 4.0f * count / num_points;
        std::cout << "Estimated value of Pi: " << pi << std::endl;
    }

    return 0;
}