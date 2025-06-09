#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "../Matrix.h"

int main() {
    std::vector<int> num_points_list = {62500000, 125000000, 250000000, 500000000, 1000000000, 2000000000}; 
    
    try {
        sycl::queue queue{sycl::gpu_selector_v};
        
        std::cout << "Running on device: " 
                  << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

        for(int &num_points : num_points_list) {
            std::cout << "Running SYCL Monte Carlo simulation with " << num_points << " points..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();

            std::vector<float> random_x(num_points);
            std::vector<float> random_y(num_points);
            
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < num_points; ++i) {
                random_x[i] = dist(rng);
                random_y[i] = dist(rng);
            }

            sycl::buffer<float, 1> buffer_x(random_x.data(), sycl::range<1>(num_points));
            sycl::buffer<float, 1> buffer_y(random_y.data(), sycl::range<1>(num_points));
            sycl::buffer<int, 1> buffer_count(sycl::range<1>(1));

            queue.submit([&](sycl::handler& h) {
                auto acc_x = buffer_x.get_access<sycl::access::mode::read>(h);
                auto acc_y = buffer_y.get_access<sycl::access::mode::read>(h);
                
                auto count_reducer = sycl::reduction(buffer_count, h, sycl::plus<int>());
                
                h.parallel_for(sycl::range<1>(num_points), 
                    count_reducer,
                    [=](sycl::id<1> idx, auto& sum) {
                        int id = idx[0];
                        float x = acc_x[id];
                        float y = acc_y[id];
                        
                        if (x * x + y * y <= 1.0f) {
                            sum += 1;
                        }
                    });
            });

            queue.wait();
            auto host_acc = buffer_count.get_host_access();
            int count = host_acc[0];

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;
            std::cout << "SYCL Monte Carlo completed in " << duration.count() << " seconds." << std::endl;
            saveDurationToFile("results/MonteCarlo/Sycl.txt", num_points, duration.count());

            float pi = 4.0f * count / num_points;
            std::cout << "Estimated value of Pi: " << pi << std::endl;
        }
    } catch (sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}