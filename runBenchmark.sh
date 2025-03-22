#!/bin/bash

echo "Run specified benchmark"
echo "1. CPU"
echo "2. CUDA"
echo "3. OpenCL"
echo "4. OpenMP"
echo "5. Sycl"
read -p "Enter (1-5): " choice

echo "Choose a benchmark to run:"
echo "1. algo1"
echo "2. algo2"
echo "3. algo3"
read -p "Enter (1-3): " algo_choice

case $choice in
    1)
        ./build/bin/Debug/cpu_benchmark.exe $algo_choice
        ;;
    2)
        ./build/bin/Debug/cuda_benchmark.exe $algo_choice
        ;;
    3)
        ./build/bin/Debug/opencl_benchmark.exe $algo_choice
        ;;
    4)
        ./build/bin/Debug/openmp_benchmark.exe $algo_choice
        ;;
    5)
        ./build/bin/Debug/sycl_benchmark.exe $algo_choice
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac