#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include "../Matrix.h"

int main() {
    std::vector<int> sizes = {33554432, 67108864, 134217728, 268435456, 536870912, 1073741824};

    for(int &n : sizes) {
        std::vector<int> input(n);
        int workGroupSize = 256, numWorkGroupSize = n / workGroupSize;

        for (auto& val : input) {
            val = rand() % 100;
        }

        std::vector<int> squared(n);
        std::vector<int> partialSums(numWorkGroupSize);

        std::cout << "Running OpenMP MapReduce style data processing..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            squared[i] = input[i] * input[i];
        }

        #pragma omp parallel for
        for (int group = 0; group < numWorkGroupSize; ++group) {
            int startIdx = group * workGroupSize;
            int endIdx = startIdx + workGroupSize;
            partialSums[group] = std::accumulate(squared.begin() + startIdx, squared.begin() + endIdx, 0);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;        
        std::cout << "OpenMP MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/MapReduce/OpenMP.txt", n, duration.count());

        int64_t result = std::accumulate(partialSums.begin(), partialSums.end(), (int64_t)0);
        std::cout << "Reduction result: " << result << std::endl;
    }

    return 0;
}