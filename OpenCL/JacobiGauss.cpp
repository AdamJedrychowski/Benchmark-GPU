#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include "../Matrix.h"

// OpenCL kernel for Jacobi Iteration
const char* jacobiKernelSource = R"(
__kernel void jacobi(
    __global const float* A,
    __global const float* b,
    __global float* x,
    __global float* x_new,
    const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                sum += A[i * n + j] * x[j];
            }
        }
        x_new[i] = (b[i] - sum) / A[i * n + i];
    }
}
)";

int main() {
    srand(static_cast<unsigned int>(time(0)));
    int n = 150;
    Matrix A = Matrix<float>::generateMatrixSystemEquations(n);
    std::vector<float> b(n);
    for (int i = 0; i < n; ++i) {
        b[i] = static_cast<float>(rand() % 1000000 + 1);
    }
    std::vector<float> x(n), x_new(n);

    int maxIterations = 1000;
    float tolerance = 1e-6;

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Device device = devices.front();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, jacobiKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "jacobi");

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n*n, A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * b.size(), b.data());
        cl::Buffer bufferX(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * x.size(), x.data());
        cl::Buffer bufferXNew(context, CL_MEM_READ_WRITE, sizeof(float) * x_new.size());

        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferX);
        kernel.setArg(3, bufferXNew);
        kernel.setArg(4, n);

        std::cout << "Starting Jacobi Iteration..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < maxIterations; ++iter) {
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n));
            queue.enqueueReadBuffer(bufferXNew, CL_TRUE, 0, sizeof(float) * x_new.size(), x_new.data());

            float error = 0.0f;
            for (int i = 0; i < n; ++i) {
                error += std::abs(x_new[i] - x[i]);
            }
            if (error < tolerance) {
                break;
            }
            queue.enqueueWriteBuffer(bufferX, CL_TRUE, 0, sizeof(float) * x.size(), x_new.data());
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "Jacobi Iteration completed in " << duration.count() << " seconds." << std::endl;
        std::cout << "Solution: ";
        for (int i = 0; i < n; ++i) {
            std::cout << x_new[i] << " ";
        }

    } catch (const cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}