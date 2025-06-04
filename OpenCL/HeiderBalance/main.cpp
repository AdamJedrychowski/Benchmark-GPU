#include <stdlib.h>
#include <filesystem>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include "CL/opencl.hpp"
#include "../Matrix.h"

void numberOfEdgesAndTriads(Matrix<int> &a, long long &links, long long &triads)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.size(); ++i)
    {
        for (int j = i + 1; j < a.size(); ++j)
        {
            links += std::abs(a(i, j));
            for (int k = j + 1; k < a.size(); ++k)
            {
                triads += std::abs(a(i, j) * a(j, k) * a(k, i));
            }
        }
    }
    std::cout << "Edges: " << links << std::endl;
    std::cout << "Triads: " << triads << std::endl;
}

void sumWorkGroupsProperties(std::vector<int> &sum_xmean, std::vector<int> &sum_energy, int links, int triads, 
                                     int numWorkGroups, int workGroupSize, std::vector<float> &xmean, std::vector<float> &energy)
{
    xmean.push_back(0.0f);
    energy.push_back(0.0f);
    for(int i=0; i< numWorkGroups; ++i) {
        xmean.back() += sum_xmean[i];
        energy.back() -= sum_energy[i];
    }
    xmean.back() /= links;
    energy.back() /= triads;
}

void saveResultsToFile(std::string prefix, int nodes, long long duration, 
                       const std::vector<float> &xmean, const std::vector<float> &energy)
{
    std::fstream out;
    try {
        out.open(prefix+"_node.csv", std::ios::app);
        out << nodes << ";" << duration << std::endl;
        out.close();

        out.open(prefix + "_mean.csv", std::ios::out);
        for (std::size_t i = 0; i < xmean.size(); ++i)
        {
            out << i << ";" << xmean[i] << std::endl;
        }
        out.close();

        out.open(prefix + "_energy.csv", std::ios::out);
        for (std::size_t i = 0; i < energy.size(); ++i)
        {
            out << i << ";" << energy[i] << std::endl;
        }
        out.close();
    } catch (const std::ios_base::failure &e) {
        std::cout << "Error writing to file: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cout << "An error occurred: " << e.what() << std::endl;
    }
}


typedef union{
	struct{
		std::uint32_t a,b,c,d;
	};
	std::uint64_t res;
} tyche_i_state;

#define TYCHE_I_ROT(a,b) (((a) >> (b)) | ((a) << (32 - (b))))
void tyche_i_advance(tyche_i_state* state){
	state->b = TYCHE_I_ROT(state->b, 7) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 8) ^ state->a;
	state->a -= state->b;
	state->b = TYCHE_I_ROT(state->b, 12) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 16) ^ state->a;
	state->a -= state->b;
}

int main() {
    // Initialization of variables
    long long links = 0, triads = 0, duration;
    int nodes = 256, t_max = 1000;
    float temperature = 5.0;
    Matrix<int> matrix = Matrix<int>::generateRandomMatrix(nodes), a(nodes);

    // Vectors and variables for results
    std::vector<float> xmean, energy;
    std::vector<int> sum_xmean, sum_energy;

    // Calculate number of edges and triads
    numberOfEdgesAndTriads(matrix, links, triads);
    std::cout << "Number of nodes: " << nodes << "\n";
    std::cout << "Number of edges: " << links << "\n";
    std::cout << "Number of triads: " << triads << "\n";

    // OpenCL file reading
    std::ifstream stream("OpenCLRun.cl");
    std::stringstream fileOpenCL;
    fileOpenCL << stream.rdbuf();

    cl::Buffer dev_x_copy, dev_sum_xmean, dev_sum_energy, dev_probability, dev_seed;
    std::vector<float> probability;
    std::vector<tyche_i_state> seed;

    try {
        // OpenCL context and program setup
        cl::Context context(CL_DEVICE_TYPE_GPU);
        cl::Program program(context, fileOpenCL.str(), true);
        cl::CommandQueue queueResults(context);
        cl::Kernel kernelProperties(program, "calculateProperties");

        // Buffers and vectors initialization
        cl::CommandQueue queueAlgorithm(context);
        cl::Kernel kernelStart(program, "startSyncSim");

        // Parallel matrix initialization
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nodes; i++) {
            for (int j = i + 1; j < nodes; ++j) {
                if (matrix(i, j)) {
                    a(i, j) = a(j, i) = 1;
                }
            }
        }

        cl::Buffer dev_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * nodes * nodes, matrix.data());
        cl::Buffer dev_x_new(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * nodes * nodes, matrix.data());
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * nodes * nodes, a.data());

        // Reserve space for vectors
        xmean.reserve(t_max + 1);
        energy.reserve(t_max + 1);
        probability.reserve(2 * nodes + 1);

        // Probability calculation
        for (int ksi = -nodes; ksi <= nodes; ++ksi) {
            probability.push_back(1.0 / (1.0 + std::exp(-2.0 * ksi / temperature)));
        }

        // Seed initialization
        seed.resize(nodes * nodes);
        int random;
        for (int i = 0; i < nodes; ++i) {
            for (int j = 0; j < nodes; ++j) {
                random = rand();
                seed[i * nodes + j].a = random >> 32;
                seed[i * nodes + j].b = random;
                seed[i * nodes + j].c = 2654435769;
                seed[i * nodes + j].d = 1367130551 ^ (i + nodes * (j + nodes));
                for (int k = 0; k < 20; ++k) {
                    tyche_i_advance(&seed[i * nodes + j]);
                }
            }
        }
        dev_seed = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(tyche_i_state) * nodes * nodes, seed.data());
        dev_probability = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * (2 * nodes + 1), probability.data());

        // Kernel argument setup
        kernelStart.setArg(0, dev_x);
        kernelStart.setArg(1, dev_x_new);
        kernelStart.setArg(2, dev_a);
        kernelStart.setArg(3, nodes);
        kernelStart.setArg(4, dev_probability);
        kernelStart.setArg(5, dev_seed);

        dev_x_copy = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * nodes * nodes);

        // Work-group calculation
        int workGroupSize = 256;
        int numWorkGroups = matrix.size() * matrix.size() / workGroupSize;
        std::cout << "Number of work-items in work-group: " << workGroupSize << "\nNumber of work-groups: " << numWorkGroups << "\n\n";
        sum_xmean.resize(numWorkGroups);
        sum_energy.resize(numWorkGroups);

        dev_sum_xmean = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * numWorkGroups);
        dev_sum_energy = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * numWorkGroups);

        // Kernel properties setup
        kernelProperties.setArg(0, dev_x_copy);
        kernelProperties.setArg(1, dev_a);
        kernelProperties.setArg(2, nodes);
        kernelProperties.setArg(3, sizeof(int) * workGroupSize, nullptr);
        kernelProperties.setArg(4, sizeof(int) * workGroupSize, nullptr);
        kernelProperties.setArg(5, dev_sum_energy);
        kernelProperties.setArg(6, dev_sum_xmean);



        std::cout << "Wait 30 secounds to calm down the processor\n";
        // std::this_thread::sleep_for(std::chrono::seconds(30));

        std::cout << "Start algorithm\n";
        auto begin = std::chrono::high_resolution_clock::now();
        for(int t=0; t<t_max; ++t) {
            queueAlgorithm.enqueueNDRangeKernel(kernelStart, cl::NullRange, cl::NDRange(nodes, nodes));

            queueResults.finish();
            queueAlgorithm.enqueueCopyBuffer(dev_x_new, dev_x_copy, 0, 0, sizeof(int) * nodes * nodes);
            sumWorkGroupsProperties(sum_xmean, sum_energy, links, triads, numWorkGroups, workGroupSize, xmean, energy);
            queueAlgorithm.finish();
            queueAlgorithm.enqueueCopyBuffer(dev_x_new, dev_x, 0, 0, sizeof(int) * nodes * nodes);

            queueResults.enqueueNDRangeKernel(kernelProperties, cl::NullRange, cl::NDRange(nodes * nodes), cl::NDRange(workGroupSize));
            queueResults.enqueueReadBuffer(dev_sum_xmean, false, 0, sizeof(int) * numWorkGroups, sum_xmean.data());
            queueResults.enqueueReadBuffer(dev_sum_energy, false, 0, sizeof(int) * numWorkGroups, sum_energy.data());
            // for(int i = 0; i < numWorkGroups; ++i) {
            //     std::cout << "Work-group " << i << ": xmean = " << sum_xmean[i] << ", energy = " << sum_energy[i] << std::endl;
            // }
        }
        queueResults.finish();
        sumWorkGroupsProperties(sum_xmean, sum_energy, links, triads, numWorkGroups, workGroupSize, xmean, energy);

        xmean.erase(xmean.begin());
        energy.erase(energy.begin());

        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    } catch (const cl::Error &e) {
        std::cout << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return -1;
    } catch (const std::exception &e) {
        std::cout << "Standard Exception: " << e.what() << std::endl;
        return -1;
    }
    saveResultsToFile("tmp/" + std::to_string(temperature), nodes, duration, xmean, energy);

    return 0;
}


