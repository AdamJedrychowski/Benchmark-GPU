#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

void printDeviceInfo() {
    try {
        // Get all platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (size_t i = 0; i < platforms.size(); ++i) {
            std::cout << "Platform " << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

            // Get all devices for the platform
            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (size_t j = 0; j < devices.size(); ++j) {
                std::cout << "  Device " << j + 1 << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << std::endl;
                std::cout << "    Compute Units: " << devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
                std::cout << "    Global Memory: " << devices[j].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MB" << std::endl;
                std::cout << "    Local Memory: " << devices[j].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB" << std::endl;
                std::cout << "    Max Work Group Size: " << devices[j].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

                size_t maxWorkGroupSize;
                devices[j].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
                std::cout << "Max workgroup size: " << maxWorkGroupSize << std::endl;

                cl_ulong localMemSize;
                devices[j].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
                std::cout << "Local memory size: " << localMemSize << " bytes" << std::endl;
            }
        }
    } catch (const cl::Error &err) {
        std::cerr << "OpenCL Error: " << err.what() << " (" << err.err() << ")" << std::endl;
    }
}

int main() {
    printDeviceInfo();
    return 0;
}