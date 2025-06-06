#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

// CUDA kernel for PageRank
__global__ void pagerank_kernel(float* inRank, float* outRank, int* nodeOffsets, int* nodeConnections, int* outDegrees, float dampingFactor, int numNodes) {
    int nodeId = blockIdx.x * blockDim.x + threadIdx.x;

    if (nodeId < numNodes) {
        float sum = 0.0f;

        // Get the range of connections for this node
        int start = nodeOffsets[nodeId];
        int end = nodeOffsets[nodeId + 1];

        // Sum contributions from incoming connections
        for (int i = start; i < end; i++) {
            int sourceNode = nodeConnections[i];
            if (outDegrees[sourceNode] > 0) {
                sum += inRank[sourceNode] / outDegrees[sourceNode];
            } else {
                sum += inRank[sourceNode] / numNodes; // Distribute evenly if no outgoing edges
            }
        }

        // Apply damping factor to compute new rank
        outRank[nodeId] = (1.0f - dampingFactor) / numNodes + dampingFactor * sum;
    }
}

class Graph {
public:
    int numNodes;
    std::vector<std::vector<int>> adjacencyList;

    Graph(int n) : numNodes(n), adjacencyList(n) {}

    void addEdge(int from, int to) {
        adjacencyList[from].push_back(to);
    }

    void convertToCSR(std::vector<int>& nodeOffsets, std::vector<int>& nodeConnections, std::vector<int>& outDegrees) {
        nodeOffsets.resize(numNodes + 1);
        outDegrees.resize(numNodes);

        int totalConnections = 0;
        for (int i = 0; i < numNodes; i++) {
            outDegrees[i] = adjacencyList[i].size();
            totalConnections += outDegrees[i];
        }
        nodeConnections.resize(totalConnections);

        nodeOffsets[0] = 0;
        for (int i = 0; i < numNodes; i++) {
            nodeOffsets[i + 1] = nodeOffsets[i] + adjacencyList[i].size();
            for (size_t j = 0; j < adjacencyList[i].size(); j++) {
                nodeConnections[nodeOffsets[i] + j] = adjacencyList[i][j];
            }
        }
    }

    void handleDanglingNodes() {
        for (int i = 0; i < numNodes; i++) {
            if (adjacencyList[i].empty()) {
                for (int j = 0; j < numNodes; j++) {
                    if (i != j) {
                        adjacencyList[i].push_back(j);
                    }
                }
            }
        }
    }
};

Graph generateRandomGraph(int numNodes, float edgeProbability) {
    Graph graph(numNodes);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            if (i != j && dis(gen) < edgeProbability) {
                graph.addEdge(i, j);
            }
        }
    }

    graph.handleDanglingNodes();
    return graph;
}

int main() {
    int numNodes = 10000;
    float edgeProbability = 0.001f;
    float dampingFactor = 0.85f;
    int maxIterations = 100;
    float tolerance = 1e-6f;

    std::cout << "Generating random graph with " << numNodes << " nodes..." << std::endl;
    Graph graph = generateRandomGraph(numNodes, edgeProbability);

    std::vector<int> nodeOffsets;
    std::vector<int> nodeConnections;
    std::vector<int> outDegrees;
    graph.convertToCSR(nodeOffsets, nodeConnections, outDegrees);

    std::vector<float> initialRank(numNodes, 1.0f / numNodes);
    std::vector<float> resultRank(numNodes);

    float *d_inRank, *d_outRank;
    int *d_nodeOffsets, *d_nodeConnections, *d_outDegrees;

    cudaMalloc(&d_inRank, sizeof(float) * numNodes);
    cudaMalloc(&d_outRank, sizeof(float) * numNodes);
    cudaMalloc(&d_nodeOffsets, sizeof(int) * (numNodes + 1));
    cudaMalloc(&d_nodeConnections, sizeof(int) * nodeConnections.size());
    cudaMalloc(&d_outDegrees, sizeof(int) * numNodes);

    cudaMemcpy(d_inRank, initialRank.data(), sizeof(float) * numNodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeOffsets, nodeOffsets.data(), sizeof(int) * (numNodes + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeConnections, nodeConnections.data(), sizeof(int) * nodeConnections.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outDegrees, outDegrees.data(), sizeof(int) * numNodes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numNodes + blockSize - 1) / blockSize;

    std::cout << "Starting CUDA PageRank computation..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < maxIterations; iter++) {
        pagerank_kernel<<<numBlocks, blockSize>>>(d_inRank, d_outRank, d_nodeOffsets, d_nodeConnections, d_outDegrees, dampingFactor, numNodes);
        cudaDeviceSynchronize();

        std::swap(d_inRank, d_outRank);
    }

    cudaMemcpy(resultRank.data(), d_inRank, sizeof(float) * numNodes, cudaMemcpyDeviceToHost);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "CUDA PageRank completed in " << duration.count() << " ms" << std::endl;

    cudaFree(d_inRank);
    cudaFree(d_outRank);
    cudaFree(d_nodeOffsets);
    cudaFree(d_nodeConnections);
    cudaFree(d_outDegrees);

    return 0;
}