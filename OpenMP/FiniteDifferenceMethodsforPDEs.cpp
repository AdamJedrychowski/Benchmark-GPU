#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>


// Parametry fizyczne i numeryczne
const double alpha = 0.01;   // współczynnik przewodzenia ciepła
const double L = 1.0;        // długość pręta
const double T = 1.0;        // czas całkowity

const int Nx = 20;           // liczba punktów przestrzennych
const int Nt = 1000;         // liczba kroków czasowych

const double dx = L / (Nx - 1);
const double dt = T / Nt;

const double r = alpha * dt / (dx * dx);  // parametr stabilności

int main() {
    std::vector<double> u(Nx);
    std::vector<double> u_next(Nx);

    // Inicjalizacja warunku początkowego
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        u[i] = sin(M_PI * x);
    }

    std::cout << "Running OpenMP heat equation solver..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Iteracje czasowe
    for (int n = 0; n < Nt; ++n) {
        // Obliczanie kolejnego kroku czasowego
        #pragma omp parallel for
        for (int i = 1; i < Nx - 1; ++i) {
            u_next[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1]);
        }

        // Aktualizacja wektora u
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) {
            u[i] = u_next[i];
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "OpenMP heat equation solver completed in " << duration.count() << " ms" << std::endl;

    std::cout << "x,u_final\n";
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        std::cout << x << "," << u[i] << "\n";
    }

    return 0;
}