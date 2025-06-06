#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <omp.h>

class Graph {
public:
    int numNodes;
    std::vector<std::vector<int>> adjacencyList;

    Graph(int n) : numNodes(n), adjacencyList(n) {}

    void addEdge(int from, int to) {
        adjacencyList[from].push_back(to);
    }

    // Convert to CSR format for efficient processing
    void convertToCSR(std::vector<int>& nodeOffsets,
                      std::vector<int>& nodeConnections,
                      std::vector<int>& outDegrees) {
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

    std::vector<float> inRank(numNodes, 1.0f / numNodes);
    std::vector<float> outRank(numNodes);

    std::cout << "Starting OpenMP PageRank computation..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < maxIterations; iter++) {
        #pragma omp parallel for
        for (int nodeId = 0; nodeId < numNodes; nodeId++) {
            float sum = 0.0f;

            int start = nodeOffsets[nodeId];
            int end = nodeOffsets[nodeId + 1];

            for (int i = start; i < end; i++) {
                int sourceNode = nodeConnections[i];
                if (outDegrees[sourceNode] > 0) {
                    sum += inRank[sourceNode] / outDegrees[sourceNode];
                } else {
                    sum += inRank[sourceNode] / numNodes;
                }
            }

            outRank[nodeId] = (1.0f - dampingFactor) / numNodes + dampingFactor * sum;
        }

        float diff = 0.0f;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < numNodes; i++) {
            diff += std::abs(outRank[i] - inRank[i]);
            inRank[i] = outRank[i];
        }

        if (diff < tolerance) {
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "OpenMP PageRank completed in " << duration.count() << " ms" << std::endl;

    return 0;
}