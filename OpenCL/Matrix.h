#pragma once
#include <vector>
#include <iostream>
#include <ostream>
#include <fstream>
#include <cmath>
#include <initializer_list>
#include <stdexcept>

template<typename T=float>
class Matrix {
public:
    Matrix() {}
    Matrix(int N) : m_N(N), tab(m_N*m_N, 0) {}
    Matrix(std::string filename) {
        std::fstream myfile(filename, std::ios::in);
        if(myfile.bad()) {
            std::cout << "File " << filename << " cannot be opened\n";
            exit(-1);
        }
        myfile >> m_N;
        tab.resize(m_N*m_N);
        for(int i = 0; i < m_N; i++) {
            for (int j = 0; j < m_N; j++) {
                myfile >> operator()(i, j);
            }
        }
        myfile.close();
    }
    Matrix(std::initializer_list<T> lst) : m_N(sqrt(lst.size())), tab(lst) {
        if(floor(m_N) != m_N) throw std::length_error("The number of columns and rows in matrix must be the same.");
    }

    T &operator()(int i, int j) {
        return tab[i * m_N + j];
    }

    T operator()(int i, int j) const {
        return tab[i * m_N + j];
    }

    int size() const {
        return m_N;
    }

    T *data() {
        return tab.data();
    }

    static Matrix generateRandomMatrix(int N) {
        Matrix<T> matrix(N);
        for(int i=0; i<N; i++) {
            for(int j=i+1; j<N; j++) {
                if(rand() / (T)(RAND_MAX) < 0.5) matrix(i, j) = matrix(j, i) = -1;
                else matrix(i, j) = matrix(j, i) = 1;
            }
        }
        return matrix;
    }

    static Matrix generateMatrixSystemEquations(int N) {
        Matrix<T> A(N);

        // Create a diagonally dominant matrix A
        for (int i = 0; i < N; ++i) {
            T rowSum = 0.0f;
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    A(i, j) = static_cast<T>(rand() % 1000 + 1);
                    rowSum += A(i, j);
                }
            }
            A(i, i) = rowSum + static_cast<T>(rand() % 1000 + 1); // Ensure diagonal dominance
        }

        return A;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix &print) {
        for(int i = 0; i < print.size(); ++i) {
            for(int j = 0; j < print.size(); ++j) {
                os << std::showpos << print(i,j) << " ";
            }
            os << std::endl << std::noshowpos;
        }
        return os;
    }

private:
    int m_N;
    std::vector<T> tab;
};