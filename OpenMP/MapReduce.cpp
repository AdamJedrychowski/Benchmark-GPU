#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <chrono>

int main() {
    // Input data
    std::vector<int> input(131072);
    int workGroupSize = 256;
    int numWorkGroupSize = input.size() / workGroupSize;

    for (auto& val : input) {
        val = rand() % 100;
    }

    std::vector<int> squared(input.size());
    std::vector<int> partialSums(numWorkGroupSize);

    std::cout << "Running OpenMP MapReduce style data processing..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Map phase: square each element
    #pragma omp parallel for
    for (int i = 0; i < input.size(); ++i) {
        squared[i] = input[i] * input[i];
    }

    // Reduce phase: sum in chunks (work groups)
    #pragma omp parallel for
    for (int group = 0; group < numWorkGroupSize; ++group) {
        int startIdx = group * workGroupSize;
        int endIdx = startIdx + workGroupSize;
        partialSums[group] = std::accumulate(squared.begin() + startIdx, squared.begin() + endIdx, 0);
    }

    // Final reduction: sum all partial sums
    int64_t result = std::accumulate(partialSums.begin(), partialSums.end(), (int64_t)0);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;

    std::cout << "OpenMP MapReduce style data processing completed in " << duration.count() << " seconds." << std::endl;
    std::cout << "Reduction result: " << result << std::endl;

    return 0;
}