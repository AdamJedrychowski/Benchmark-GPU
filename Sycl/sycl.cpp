#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

void printDeviceInfo(const sycl::device& device, int deviceIndex) {
    std::cout << "\n=== Device " << deviceIndex << " ===\n";
    
    try {
        // Basic device information
        std::cout << "Name: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
        std::cout << "Driver Version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
        std::cout << "Version: " << device.get_info<sycl::info::device::version>() << std::endl;
        
        // Device type
        auto device_type = device.get_info<sycl::info::device::device_type>();
        std::cout << "Type: ";
        switch (device_type) {
            case sycl::info::device_type::cpu:
                std::cout << "CPU";
                break;
            case sycl::info::device_type::gpu:
                std::cout << "GPU";
                break;
            case sycl::info::device_type::accelerator:
                std::cout << "Accelerator";
                break;
            case sycl::info::device_type::custom:
                std::cout << "Custom";
                break;
            case sycl::info::device_type::automatic:
                std::cout << "Automatic";
                break;
            case sycl::info::device_type::host:
                std::cout << "Host";
                break;
            case sycl::info::device_type::all:
                std::cout << "All";
                break;
        }
        std::cout << std::endl;
        
        // Memory information
        std::cout << "Global Memory Size: " 
                  << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024) 
                  << " MB" << std::endl;
        std::cout << "Local Memory Size: " 
                  << device.get_info<sycl::info::device::local_mem_size>() / 1024 
                  << " KB" << std::endl;
        
        // Compute capabilities
        std::cout << "Max Compute Units: " 
                  << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
        std::cout << "Max Work Group Size: " 
                  << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
        
        // Fixed: Removed problematic max_work_item_sizes call
        std::cout << "Max Work Item Dimensions: " 
                  << device.get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
        
        // Clock and performance
        std::cout << "Max Clock Frequency: " 
                  << device.get_info<sycl::info::device::max_clock_frequency>() 
                  << " MHz" << std::endl;
        
        // Memory alignment
        std::cout << "Memory Base Address Alignment: " 
                  << device.get_info<sycl::info::device::mem_base_addr_align>() << std::endl;
        
        // Platform information
        auto platform = device.get_platform();
        std::cout << "Platform Name: " << platform.get_info<sycl::info::platform::name>() << std::endl;
        std::cout << "Platform Vendor: " << platform.get_info<sycl::info::platform::vendor>() << std::endl;
        std::cout << "Platform Version: " << platform.get_info<sycl::info::platform::version>() << std::endl;
        
        // Additional device capabilities
        std::cout << "Address Bits: " 
                  << device.get_info<sycl::info::device::address_bits>() << std::endl;
        
        // Removed deprecated endian_little
        
        std::cout << "Error Correction Support: " 
                  << (device.get_info<sycl::info::device::error_correction_support>() ? "Yes" : "No") << std::endl;
        
        // Use modern aspect-based queries instead of deprecated properties
        std::cout << "USM Device Allocations: " 
                  << (device.has(sycl::aspect::usm_device_allocations) ? "Yes" : "No") << std::endl;
        std::cout << "USM Host Allocations: " 
                  << (device.has(sycl::aspect::usm_host_allocations) ? "Yes" : "No") << std::endl;
        std::cout << "USM Shared Allocations: " 
                  << (device.has(sycl::aspect::usm_shared_allocations) ? "Yes" : "No") << std::endl;
        
        // Use modern aspect query for image support
        std::cout << "Image Support: " 
                  << (device.has(sycl::aspect::image) ? "Yes" : "No") << std::endl;
        
        // Preferred vector widths
        std::cout << "Preferred Vector Width (char): " 
                  << device.get_info<sycl::info::device::preferred_vector_width_char>() << std::endl;
        std::cout << "Preferred Vector Width (int): " 
                  << device.get_info<sycl::info::device::preferred_vector_width_int>() << std::endl;
        std::cout << "Preferred Vector Width (float): " 
                  << device.get_info<sycl::info::device::preferred_vector_width_float>() << std::endl;
        std::cout << "Preferred Vector Width (double): " 
                  << device.get_info<sycl::info::device::preferred_vector_width_double>() << std::endl;
        
        // Device aspects (modern SYCL 2020 way to query capabilities)
        std::cout << "Device Aspects:" << std::endl;
        std::cout << "  - FP16: " << (device.has(sycl::aspect::fp16) ? "Yes" : "No") << std::endl;
        std::cout << "  - FP64: " << (device.has(sycl::aspect::fp64) ? "Yes" : "No") << std::endl;
        std::cout << "  - Atomic64: " << (device.has(sycl::aspect::atomic64) ? "Yes" : "No") << std::endl;
        std::cout << "  - Queue Profiling: " << (device.has(sycl::aspect::queue_profiling) ? "Yes" : "No") << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error getting device info: " << e.what() << std::endl;
    }
}

int main() {
    try {
        std::cout << "SYCL Device Information\n";
        std::cout << "======================\n";
        
        // Get all platforms
        auto platforms = sycl::platform::get_platforms();
        std::find_if(platforms.begin(), platforms.end(), [](const sycl::platform& p) {
            return p.get_info<sycl::info::platform::name>().find("NVIDIA") != std::string::npos;
        });
        std::cout << "Found " << platforms.size() << " platform(s)\n";
        
        int deviceCount = 0;
        
        // Iterate through all platforms
        for (size_t i = 0; i < platforms.size(); ++i) {
            std::cout << "\n--- Platform " << i << " ---\n";
            std::cout << "Name: " << platforms[i].get_info<sycl::info::platform::name>() << std::endl;
            std::cout << "Vendor: " << platforms[i].get_info<sycl::info::platform::vendor>() << std::endl;
            std::cout << "Version: " << platforms[i].get_info<sycl::info::platform::version>() << std::endl;
            
            // Get all devices for this platform
            auto devices = platforms[i].get_devices();
            std::cout << "Number of devices: " << devices.size() << std::endl;
            
            // Print information for each device
            for (const auto& device : devices) {
                printDeviceInfo(device, deviceCount++);
            }
        }
        
        if (deviceCount == 0) {
            std::cout << "No SYCL devices found!" << std::endl;
        } else {
            std::cout << "\nTotal devices found: " << deviceCount << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}