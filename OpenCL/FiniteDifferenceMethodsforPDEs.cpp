#define _USE_MATH_DEFINES
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cmath>


// Parametry fizyczne i numeryczne
const double alpha = 0.01;   // współczynnik przewodzenia ciepła
const double L = 1.0;        // długość pręta
const double T = 1.0;        // czas całkowity

const int Nx = 20;           // liczba punktów przestrzennych
const int Nt = 1000;         // liczba kroków czasowych

const double dx = L / (Nx - 1);
const double dt = T / Nt;

const double r = alpha * dt / (dx * dx);  // parametr stabilności

const char* kernelSource = R"(
__kernel void heatEquation(__global double* u, __global double* u_next, const double r, const int Nx) {
    int i = get_global_id(0);
    if (i > 0 && i < Nx - 1) {
        u_next[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1]);
    }
}
)";

int main() {
    std::vector<double> u(Nx);
    std::vector<double> u_next(Nx);

    // Inicjalizacja warunku początkowego
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        u[i] = sin(M_PI * x);
    }

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

        cl::Buffer buffer_u(context, CL_MEM_READ_WRITE, sizeof(double) * Nx);
        cl::Buffer buffer_u_next(context, CL_MEM_READ_WRITE, sizeof(double) * Nx);

        queue.enqueueWriteBuffer(buffer_u, CL_TRUE, 0, sizeof(double) * Nx, u.data());

        cl::Kernel kernel(program, "heatEquation");
        kernel.setArg(0, buffer_u);
        kernel.setArg(1, buffer_u_next);
        kernel.setArg(2, r);
        kernel.setArg(3, Nx);

        std::cout << "Running OpenCL heat equation solver..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int n = 0; n < Nt; ++n) {
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(Nx), cl::NullRange);
            queue.enqueueCopyBuffer(buffer_u_next, buffer_u, 0, 0, sizeof(double) * Nx);
        }

        queue.enqueueReadBuffer(buffer_u, CL_TRUE, 0, sizeof(double) * Nx, u.data());

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "OpenCL heat equation solver completed in " << duration.count() << " ms" << std::endl;

        std::cout << "x,u_final\n";
        for (int i = 0; i < Nx; ++i) {
            double x = i * dx;
            std::cout << x << "," << u[i] << "\n";
        }

    } catch (cl::Error& e) {
        std::cout << "OpenCL error: " << e.what() << "(" << e.err() << ")" << std::endl;
        return 1;
    }

    return 0;
}