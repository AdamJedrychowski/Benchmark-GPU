#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <openacc.h>
#include <chrono>
#include "../Matrix.h"

int main() {
    std::vector<int> sizes = {33554432, 67108864, 134217728, 268435456, 536870912, 1073741824};

    for(int &n : sizes) {
        std::vector<int> input(n);
        for (auto& val : input) {
            val = rand() % 100;
        }

        std::vector<int> squared(n);
        std::cout << "Running OpenACC MapReduce style data processing..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        // Convert vectors to raw arrays for better OpenACC compatibility
        int* input_ptr = input.data();
        int* squared_ptr = squared.data();

        // Map phase: Square each element of the input array
        #pragma acc parallel loop copyin(input_ptr[0:n]) copyout(squared_ptr[0:n])
        for (int i = 0; i < n; i++) {
            squared_ptr[i] = input_ptr[i] * input_ptr[i];
        }

        int64_t sum = 0;
        #pragma acc parallel loop reduction(+:sum) copyin(squared_ptr[0:n])
        for (int i = 0; i < n; i++) {
            sum += squared_ptr[i];
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;        
        std::cout << "OpenACC MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/MapReduce/OpenACC.txt", n, duration.count());

        std::cout << "Reduction result: " << sum << std::endl;
    }

    return 0;
}