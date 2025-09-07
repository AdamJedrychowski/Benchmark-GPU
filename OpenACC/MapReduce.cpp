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
        int workGroupSize = 256, numWorkGroupSize = n / workGroupSize;

        for (auto& val : input) {
            val = rand() % 100;
        }

        std::vector<int> squared(n);
        std::vector<int> partialSums(numWorkGroupSize);

        std::cout << "Running OpenACC MapReduce style data processing..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        #pragma acc data copyin(input[0:n]) create(squared[0:n], partialSums[0:numWorkGroupSize])
        {
            #pragma acc parallel loop
            for (int i = 0; i < n; ++i) {
                squared[i] = input[i] * input[i];
            }

            #pragma acc parallel loop
            for (int group = 0; group < numWorkGroupSize; ++group) {
                int startIdx = group * workGroupSize;
                int endIdx = startIdx + workGroupSize;
                int sum = 0;
                #pragma acc loop seq
                for (int i = startIdx; i < endIdx; ++i) {
                    sum += squared[i];
                }
                partialSums[group] = sum;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;        
        std::cout << "OpenACC MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;
        saveDurationToFile("results/MapReduce/OpenACC.txt", n, duration.count());

        int64_t result = std::accumulate(partialSums.begin(), partialSums.end(), (int64_t)0);
        std::cout << "Reduction result: " << result << std::endl;
    }

    return 0;
}