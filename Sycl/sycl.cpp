#include <sycl/sycl.hpp>

int main() {
    // Use SYCL to verify it's working properly
    sycl::queue q;
    
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "SYCL setup successful!" << std::endl;
    
    return 0;
}