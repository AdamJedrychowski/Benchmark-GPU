#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <sycl/sycl.hpp>
#include "../Matrix.h"


class Graph {
public:
    int numNodes;
    std::vector<std::vector<int>> adjacencyList;
    
    Graph(int n) : numNodes(n), adjacencyList(n) {}
    
    void addEdge(int from, int to) {
        adjacencyList[from].push_back(to);
    }
    
    // Convert to CSR format for efficient GPU processing
    void convertToCSR(std::vector<int>& nodeOffsets,
                      std::vector<int>& nodeConnections,
                      std::vector<int>& outDegrees) {
        nodeOffsets.resize(numNodes + 1);
        outDegrees.resize(numNodes);
        
        // Count total connections to size nodeConnections
        int totalConnections = 0;
        for (int i = 0; i < numNodes; i++) {
            outDegrees[i] = adjacencyList[i].size();
            totalConnections += outDegrees[i];
        }
        nodeConnections.resize(totalConnections);
        
        // Fill CSR format
        nodeOffsets[0] = 0;
        for (int i = 0; i < numNodes; i++) {
            nodeOffsets[i + 1] = nodeOffsets[i] + adjacencyList[i].size();
            for (size_t j = 0; j < adjacencyList[i].size(); j++) {
                nodeConnections[nodeOffsets[i] + j] = adjacencyList[i][j];
            }
        }
    } 
    
    // Add connections for dangling nodes (nodes with no outgoing edges)
    void handleDanglingNodes() {
        for (int i = 0; i < numNodes; i++) {
            if (adjacencyList[i].empty()) {
                // Connect dangling nodes to all other nodes
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
    // Parameters
    float edgeProbability = 0.001f;
    float dampingFactor = 0.85f;
    int maxIterations = 1000;
    
    try {
        // Create SYCL queue with GPU selector
        sycl::queue q(sycl::gpu_selector{});
        
        std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        std::vector<int> nodes = {2500, 5000, 10000, 20000, 40000, 80000};

        for(int &numNodes : nodes) {
            std::cout << "Generating random graph with " << numNodes << " nodes..." << std::endl;
            Graph graph = generateRandomGraph(numNodes, edgeProbability);

            // Convert to CSR format
            std::vector<int> nodeOffsets;
            std::vector<int> nodeConnections;
            std::vector<int> outDegrees;
            graph.convertToCSR(nodeOffsets, nodeConnections, outDegrees);

            // Initialize rank arrays
            std::vector<float> initialRank(numNodes, 1.0f / numNodes);
            std::vector<float> resultRank(numNodes);

            // Create SYCL buffers
            sycl::buffer<float, 1> inRankBuf(initialRank.data(), sycl::range<1>(numNodes));
            sycl::buffer<float, 1> outRankBuf(resultRank.data(), sycl::range<1>(numNodes));
            sycl::buffer<int, 1> nodeOffsetsBuf(nodeOffsets.data(), sycl::range<1>(numNodes + 1));
            sycl::buffer<int, 1> nodeConnectionsBuf(nodeConnections.data(), sycl::range<1>(nodeConnections.size()));
            sycl::buffer<int, 1> outDegreesBuf(outDegrees.data(), sycl::range<1>(numNodes));

            std::cout << "Starting SYCL PageRank computation..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < maxIterations; iter++) {
                // Submit PageRank kernel
                q.submit([&](sycl::handler& h) {
                    auto inRank = inRankBuf.get_access<sycl::access::mode::read>(h);
                    auto outRank = outRankBuf.get_access<sycl::access::mode::write>(h);
                    auto nodeOffsets = nodeOffsetsBuf.get_access<sycl::access::mode::read>(h);
                    auto nodeConnections = nodeConnectionsBuf.get_access<sycl::access::mode::read>(h);
                    auto outDegrees = outDegreesBuf.get_access<sycl::access::mode::read>(h);

                    h.parallel_for(sycl::range<1>(numNodes), [=](sycl::id<1> idx) {
                        int nodeId = idx[0];
                        float sum = 0.0f;
                        
                        // Get the range of connections for this node
                        int start = nodeOffsets[nodeId];
                        int end = nodeOffsets[nodeId + 1];
                        
                        // Sum contributions from incoming connections
                        for (int i = start; i < end; i++) {
                            int sourceNode = nodeConnections[i];
                            // Handle nodes with zero outgoing edges safely
                            if (outDegrees[sourceNode] > 0) {
                                sum += inRank[sourceNode] / outDegrees[sourceNode];
                            } else {
                                sum += inRank[sourceNode] / numNodes;
                            }
                        }
                        
                        // Apply damping factor to compute new rank
                        outRank[nodeId] = (1.0f - dampingFactor) / numNodes + dampingFactor * sum;
                    });
                });

                // Copy output to input for next iteration
                q.submit([&](sycl::handler& h) {
                    auto inRank = inRankBuf.get_access<sycl::access::mode::write>(h);
                    auto outRank = outRankBuf.get_access<sycl::access::mode::read>(h);

                    h.parallel_for(sycl::range<1>(numNodes), [=](sycl::id<1> idx) {
                        inRank[idx] = outRank[idx];
                    });
                });
            }

            // Wait for all operations to complete
            q.wait();

            // Read back results
            {
                auto inRankAcc = inRankBuf.get_access<sycl::access::mode::read>();
                for (int i = 0; i < numNodes; i++) {
                    resultRank[i] = inRankAcc[i];
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;

            std::cout << "SYCL PageRank completed in " << duration.count() << " seconds." << std::endl;
            saveDurationToFile("results/PageRank/Sycl.txt", numNodes, duration.count());
        }
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}