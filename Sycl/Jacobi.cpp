#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "../Matrix.h"

int main() {
    srand(1000);
    try {
        sycl::queue queue{sycl::gpu_selector{}};
        
        std::cout << "Running on device: " 
                  << queue.get_device().get_info<sycl::info::device::name>() 
                  << std::endl;

        int maxIterations = 1000;
        float tolerance = 1e-6;
        std::vector<int> sizes = {1250, 2500, 5000, 10000, 20000}; //, 40000

        for(int &n : sizes) {
            Matrix<float> A(n);
            Matrix<float>::generateMatrixSystemEquations(A);
            std::vector<float> b(n);
            for (int i = 0; i < n; ++i) {
                b[i] = static_cast<float>(rand() % 1000000 + 1);
            }
            std::vector<float> x(n), x_new(n);

            sycl::buffer<float, 2> bufferA(A.data(), sycl::range<2>(n, n));
            sycl::buffer<float, 1> bufferB(b.data(), sycl::range<1>(n));
            sycl::buffer<float, 1> bufferX(x.data(), sycl::range<1>(n));
            sycl::buffer<float, 1> bufferXNew(x_new.data(), sycl::range<1>(n));

            std::cout << "Starting Jacobi Iteration for size " << n << "..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < maxIterations; ++iter) {
                queue.submit([&](sycl::handler& h) {
                    auto A_acc = bufferA.get_access<sycl::access::mode::read>(h);
                    auto b_acc = bufferB.get_access<sycl::access::mode::read>(h);
                    auto x_acc = bufferX.get_access<sycl::access::mode::read>(h);
                    auto x_new_acc = bufferXNew.get_access<sycl::access::mode::write>(h);

                    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                        float sum = 0.0f;
                        for (int j = 0; j < n; ++j) {
                            if (i != j) {
                                sum += A_acc[i][j] * x_acc[j];
                            }
                        }
                        x_new_acc[i] = (b_acc[i] - sum) / A_acc[i][i];
                    });
                });
                queue.wait();

                float error = 0.0f;
                {
                    auto x_acc = bufferX.get_access<sycl::access::mode::read>();
                    auto x_new_acc = bufferXNew.get_access<sycl::access::mode::read>();
                    
                    for (int i = 0; i < n; ++i) {
                        error += std::abs(x_new_acc[i] - x_acc[i]);
                    }
                }

                if (error < tolerance) {
                    std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
                    break;
                }

                queue.submit([&](sycl::handler& h) {
                    auto x_acc = bufferX.get_access<sycl::access::mode::write>(h);
                    auto x_new_acc = bufferXNew.get_access<sycl::access::mode::read>(h);
                    h.copy(x_new_acc, x_acc);
                });
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;
            std::cout << "Jacobi Iteration completed in " << duration.count() << " seconds." << std::endl;
            saveDurationToFile("results/Jacobi/Sycl.txt", n, duration.count());
        }
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}