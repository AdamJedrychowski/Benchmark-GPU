@echo off
pushd %~dp0

echo Run specified benchmark
echo 1. CPU
echo 2. CUDA
echo 3. OpenCL
echo 4. OpenMP
echo 5. Sycl
set /p choice="Enter (1-5): "

echo Choose a benchmark to run:
echo 1. algo1
echo 2. algo2
echo 3. algo3
set /p algo_choice="Enter (1-3): "

if "%choice%"=="1" (
    .\build\bin\Debug\cpu_benchmark.exe %algo_choice%
) else if "%choice%"=="2" (
    .\build\bin\Debug\cuda_benchmark.exe %algo_choice%
) else if "%choice%"=="3" (
    .\build\bin\Debug\opencl_benchmark.exe %algo_choice%
) else if "%choice%"=="4" (
    .\build\bin\Debug\openmp_benchmark.exe %algo_choice%
) else if "%choice%"=="5" (
    .\build\bin\Debug\sycl_benchmark.exe %algo_choice%
) else (
    echo Invalid choice. Please run the script again.
)