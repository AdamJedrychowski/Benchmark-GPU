#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "../Matrix.h"

int main() {
    try {
        sycl::queue queue{sycl::gpu_selector{}};
        std::cout << "Running on device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::vector<int> sizes = {33554432, 67108864, 134217728, 268435456, 536870912, 1073741824};

        for(int &n : sizes) {
            std::vector<int> input(n);
            int workGroupSize = 256;
            int numWorkGroups = n / workGroupSize;
            
            for(auto& val : input) {
                val = rand() % 100;
            }

            sycl::buffer<int, 1> inputBuffer{input.data(), sycl::range<1>(n)};
            sycl::buffer<int, 1> outputBuffer{sycl::range<1>(n)};
            sycl::buffer<int, 1> resultBuffer{sycl::range<1>(numWorkGroups)};

            std::cout << "Running SYCL MapReduce style data processing..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler& h) {
                auto input_acc = inputBuffer.get_access<sycl::access::mode::read>(h);
                auto output_acc = outputBuffer.get_access<sycl::access::mode::write>(h);
                
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    output_acc[id] = input_acc[id] * input_acc[id];
                });
            });

            queue.submit([&](sycl::handler& h) {
                auto data_acc = outputBuffer.get_access<sycl::access::mode::read_write>(h);
                auto result_acc = resultBuffer.get_access<sycl::access::mode::write>(h);
                
                sycl::local_accessor<int, 1> localData(sycl::range<1>(workGroupSize), h);
                
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(workGroupSize)), 
                    [=](sycl::nd_item<1> item) {
                        int id = item.get_global_id(0);
                        int localId = item.get_local_id(0);
                        int groupSize = item.get_local_range(0);
                        
                        localData[localId] = data_acc[id];
                        item.barrier(sycl::access::fence_space::local_space);
                        
                        int prevOffset = groupSize;
                        for (int offset = groupSize / 2; offset > 0; offset /= 2) {
                            if (localId < offset) {
                                if(localId == offset - 1 && offset*2 != prevOffset) {
                                    localData[localId] += localData[localId + offset + 1];
                                }
                                localData[localId] += localData[localId + offset];
                                prevOffset = offset;
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                        
                        if (localId == 0) {
                            result_acc[item.get_group(0)] = localData[0];
                        }
                    });
            });
            queue.wait();
            
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;
            
            auto result_acc = resultBuffer.get_access<sycl::access::mode::read>();
            std::vector<int> results(numWorkGroups);
            for(int i = 0; i < numWorkGroups; i++) {
                results[i] = result_acc[i];
            }
            
            std::cout << "SYCL MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;
            saveDurationToFile("results/MapReduce/Sycl.txt", n, duration.count());

            int64_t result = std::accumulate(results.begin(), results.end(), (int64_t)0);
            std::cout << "Reduction result: " << result << std::endl;
        }
    } catch (sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}