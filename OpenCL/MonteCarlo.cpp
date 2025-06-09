#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "../Matrix.h"

const char* kernelSource = R"(
__kernel void monte_carlo_pi(__global const float* random_x, __global const float* random_y, __global int* count, const int num_points) {
    int id = get_global_id(0);
    if (id < num_points) {
        float x = random_x[id];
        float y = random_y[id];
        if (x * x + y * y <= 1.0f) {
            atomic_inc(count);
        }
    }
}
)";

int main(int argc, char** argv) {
    std::vector<int> num_points_list = {62500000, 125000000, 250000000, 500000000, 1000000000, 2000000000}; 
    
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
        cl::Kernel kernel(program, "monte_carlo_pi");

        for(int &num_points : num_points_list) {
            std::vector<float> random_x(num_points);
            std::vector<float> random_y(num_points);
            int count = 0;

            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < num_points; ++i) {
                random_x[i] = dist(rng);
                random_y[i] = dist(rng);
            }

            cl::Buffer buffer_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_points, random_x.data());
            cl::Buffer buffer_y(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_points, random_y.data());
            cl::Buffer buffer_count(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &count);

            std::cout << "Running OpenCL Monte Carlo simulation to estimate Pi..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
        
            kernel.setArg(0, buffer_x);
            kernel.setArg(1, buffer_y);
            kernel.setArg(2, buffer_count);
            kernel.setArg(3, num_points);

            cl::NDRange global(num_points);
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
            queue.enqueueReadBuffer(buffer_count, CL_TRUE, 0, sizeof(int), &count);

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;
            std::cout << "OpenCL Monte Carlo completed in " << duration.count() << " seconds." << std::endl;
            saveDurationToFile("results/MonteCarlo/OpenCL.txt", num_points, duration.count());

            float pi = 4.0f * count / num_points;
            std::cout << "Estimated value of Pi: " << pi << std::endl;
        }
    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "(" << e.err() << ")" << std::endl;
        return 1;
    }

    return 0;
}