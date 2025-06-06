#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include "../Matrix.h"

int main() {
    srand(static_cast<unsigned int>(time(0)));

    const int N = 256;
    Matrix A = Matrix<float>::generateMatrixSystemEquations(N);
    std::vector<float> b(N);
    for (int i = 0; i < N; ++i) {
        b[i] = static_cast<float>(rand() % 1000000 + 1);
    }
    std::vector<float> x(N, 0.0f);
    std::vector<float> r(N, 0.0f), p(N, 0.0f), Ap(N, 0.0f);

    // Compute initial r and p values
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A(i, j) * x[j];
        }
        p[i] = r[i] = b[i] - sum;
    }

    double rsold = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    try {
        sycl::queue queue(sycl::gpu_selector_v);
        std::cout << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
        auto startTime = std::chrono::high_resolution_clock::now();

        sycl::buffer<float, 2> bufferA(A.data(), sycl::range<2>(N, N));
        sycl::buffer<float> bufferP(p.data(), sycl::range<1>(N));
        sycl::buffer<float> bufferAp(Ap.data(), sycl::range<1>(N));

        for (int iter = 0; iter < N; ++iter) {
            queue.submit([&](sycl::handler& cgh) {
                auto accA = bufferA.get_access<sycl::access::mode::read>(cgh);
                auto accP = bufferP.get_access<sycl::access::mode::read>(cgh);
                auto accAp = bufferAp.get_access<sycl::access::mode::write>(cgh);
                cgh.parallel_for<class conjugateGradient>(sycl::range<1>(N), [=](sycl::id<1> gid) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        sum += accA[gid][j] * accP[j];
                    }
                    accAp[gid] = sum;
                });
            });

            queue.wait_and_throw();
            auto hostAp = bufferAp.get_access<sycl::access::mode::read>();
            double denom = std::inner_product(p.begin(), p.end(), hostAp.get_pointer(), 0.0);

            if (std::fabs(denom) < std::numeric_limits<double>::epsilon()) {
                std::cout << "Denominator too small, stopping iteration." << std::endl;
                break;
            }

            double alpha = rsold / denom;
            for (int i = 0; i < N; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * hostAp[i];
            }

            double rsnew = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
            if (std::sqrt(rsnew) < 1e-6) {
                std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
                break;
            }

            double beta = rsnew / rsold;
            for (int i = 0; i < N; ++i) {
                p[i] = r[i] + beta * p[i];
            }

            rsold = rsnew;
            // Update bufferP with new p values
            queue.submit([&](sycl::handler& cgh) {
                auto accP = bufferP.get_access<sycl::access::mode::write>(cgh);
                cgh.copy(p.data(), accP);
            });
            queue.wait_and_throw();
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Conjugate Gradient completed in " << elapsed.count() << " seconds." << std::endl;

    } catch (const sycl::exception& e) {
        std::cout << "SYCL Exception: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}