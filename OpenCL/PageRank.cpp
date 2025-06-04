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
#include <CL/opencl.hpp>

// OpenCL kernel as a string
const char* pageRankKernelSource = R"(
__kernel void pagerank_kernel(__global float* inRank,
                             __global float* outRank,
                             __global int* nodeOffsets,
                             __global int* nodeConnections,
                             __global int* outDegrees,
                             const float dampingFactor,
                             const int numNodes) {
    int nodeId = get_global_id(0);
    
    if (nodeId < numNodes) {
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
                sum += inRank[sourceNode] / numNodes; // Distribute evenly if no outgoing edges
            }
        }
        
        // Apply damping factor to compute new rank
        outRank[nodeId] = (1.0f - dampingFactor) / numNodes + dampingFactor * sum;
    }
}
)";

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

// Generate a random graph for testing
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
    
    // Ensure there are no dangling nodes which can cause numerical issues
    graph.handleDanglingNodes();
    
    return graph;
}

int main() {
    // Parameters
    int numNodes = 10000;
    float edgeProbability = 0.001f;
    float dampingFactor = 0.85f;
    int maxIterations = 100;
    float tolerance = 1e-6f;

    // Generate random graph
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

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Device device = devices.front();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Build the program
        cl::Program program(context, pageRankKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "pagerank_kernel");

        // Create buffers
        cl::Buffer inRankBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * numNodes, initialRank.data());
        cl::Buffer outRankBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * numNodes);
        cl::Buffer nodeOffsetsBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (numNodes + 1), nodeOffsets.data());
        cl::Buffer nodeConnectionsBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * nodeConnections.size(), nodeConnections.data());
        cl::Buffer outDegreesBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numNodes, outDegrees.data());
        
        kernel.setArg(2, nodeOffsetsBuffer);
        kernel.setArg(3, nodeConnectionsBuffer);
        kernel.setArg(4, outDegreesBuffer);
        kernel.setArg(5, dampingFactor);
        kernel.setArg(6, numNodes);

        std::cout << "Starting OpenCL PageRank computation..." << std::endl;

        // Measure OpenCL PageRank execution time
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < maxIterations; iter++) {
            kernel.setArg(0, inRankBuffer);
            kernel.setArg(1, outRankBuffer);

            // Execute kernel
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numNodes), cl::NullRange);
            queue.finish();

            // Swap input and output buffers for next iteration
            std::swap(inRankBuffer, outRankBuffer);
        }

        queue.enqueueReadBuffer(inRankBuffer, CL_TRUE, 0, sizeof(float) * numNodes, resultRank.data());

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        std::cout << "OpenCL PageRank completed in " << duration.count() << " ms" << std::endl;

    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "(" << e.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}