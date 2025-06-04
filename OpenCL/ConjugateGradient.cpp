#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include "Matrix.h"

// Conjugate Gradient kernel as a string
const char* kernelSource = R"(
__kernel void conjugateGradient(
    __global const float* A,
    __global const float* b,
    __global float* x,
    __global float* p,
    __global float* Ap,
    const int N) {

    int gid = get_global_id(0);

    if (gid < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[gid * N + j] * p[j];
        }
        Ap[gid] = sum;
    }
}
)";

int main() {
    srand(static_cast<unsigned int>(time(0)));
    // Example data
    // Matrix A = {4,1,2,0,
    //             1,3,0,1,
    //             2,0,3,1,
    //             0,1,1,2};

    const int N = 1500;
    Matrix A = Matrix::generateMatrixSystemEquations(N);
    std::vector<float> b(N);
    for (int i = 0; i < N; ++i) {
        b[i] = static_cast<float>(rand() % 1000000 + 1);
    }
    std::vector<float> x(N, 0);

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Device device = devices.front();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, kernelSource);
        program.build({device});

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, b.data());
        cl::Buffer bufferX(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, x.data());
        cl::Buffer bufferP(context, CL_MEM_READ_WRITE, sizeof(float) * N);
        cl::Buffer bufferAp(context, CL_MEM_READ_WRITE, sizeof(float) * N);

        // Initialize r and p
        std::vector<float> r(N, 0.0f);
        std::vector<float> p(N, 0.0f);
        std::vector<float> Ap(N, 0.0f);

        // Compute initial r and p values
        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += A(i, j) * x[j];
            }
            p[i] = r[i] = b[i] - sum;
        }

        // Kernel setup
        cl::Kernel kernel(program, "conjugateGradient");
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferX);
        kernel.setArg(3, bufferP);
        kernel.setArg(4, bufferAp);
        kernel.setArg(5, N);

        double rsold = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

        std::cout << "Starting Conjugate Gradient method..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        // Main loop
        for (int iter = 0; iter < N; ++iter) {
            queue.enqueueWriteBuffer(bufferP, CL_TRUE, 0, sizeof(float) * N, p.data());

            // Launch kernel
            cl::NDRange global(N);
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

            // Read back Ap
            queue.enqueueReadBuffer(bufferAp, CL_TRUE, 0, sizeof(float) * N, Ap.data());

            // Compute alpha
            double denom = std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);
            if (std::fabs(denom) < std::numeric_limits<double>::epsilon()) {
                std::cerr << "Denominator too small, stopping iteration." << std::endl;
                break;
            }
            double alpha = rsold / denom;

            // Update x and r
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
            for (int i = 0; i < N; ++i) {
                p[i] = r[i] + beta * p[i];
            }

            rsold = rsnew;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Conjugate Gradient completed in " << elapsed.count() << " seconds." << std::endl;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Solution: ";
    for (float xi : x) {
        std::cout << xi << " ";
    }
    std::cout << std::endl;

    return 0;
}