#define _USE_MATH_DEFINES
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "../Matrix.h"

const char* kernelSource = R"(
__kernel void heatEquation(__global double* u, __global double* u_next, const double r, const int Nx) {
    int i = get_global_id(0);
    if (i > 0 && i < Nx - 1) {
        u_next[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1]);
    }
})";

int main() {
    const double alpha = 0.01; // Heat conduction coefficient
    const double L = 1.0;        // Length of the rod
    const double T = 1.0;        // Total simulation time
    const int Nx = 20;           // Grid sizes
    const double dx = L / (Nx - 1);

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
        cl::Kernel kernel(program, "heatEquation");
        
        // Number of time steps
        std::vector<int> Nt_values = {50000, 100000, 200000, 400000, 800000, 1600000};
        
        for(int & Nt : Nt_values) {
            const double dt = T / Nt;
            const double r = alpha * dt / (dx * dx);  // Stability condition
            std::vector<double> u(Nx);
            std::vector<double> u_next(Nx);

            for (int i = 0; i < Nx; ++i) {
                double x = i * dx;
                u[i] = sin(M_PI * x);
            }

            cl::Buffer buffer_u(context, CL_MEM_READ_WRITE, sizeof(double) * Nx);
            cl::Buffer buffer_u_next(context, CL_MEM_READ_WRITE, sizeof(double) * Nx);

            queue.enqueueWriteBuffer(buffer_u, CL_TRUE, 0, sizeof(double) * Nx, u.data());

            kernel.setArg(0, buffer_u);
            kernel.setArg(1, buffer_u_next);
            kernel.setArg(2, r);
            kernel.setArg(3, Nx);

            std::cout << "Running OpenCL heat equation solver..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();

            for (int n = 0; n < Nt; ++n) {
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(Nx));
                queue.enqueueCopyBuffer(buffer_u_next, buffer_u, 0, 0, sizeof(double) * Nx);
            }

            queue.enqueueReadBuffer(buffer_u, CL_TRUE, 0, sizeof(double) * Nx, u.data());

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;
            std::cout << "OpenCL heat equation solver completed in " << duration.count() << " seconds." << std::endl;

            saveDurationToFile("results/FiniteDifferenceMethodsforPDEs/OpenCL.txt", Nt, duration.count());
        }
    } catch (cl::Error& e) {
        std::cout << "OpenCL error: " << e.what() << "(" << e.err() << ")" << std::endl;
        return 1;
    }

    return 0;
}