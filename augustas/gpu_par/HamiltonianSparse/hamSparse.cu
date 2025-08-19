#include <cuda_runtime.h>
#include <cusparse.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

const double sigma = 2.099518748076819;
const double eps = 4.642973059984742;
const double mu = 1.0;
const int N = 10000000;
const int threads = 1024;
const double rmin = 2.0;
const double rmax = 2.7;
const double dr = (rmax - rmin) / (N);


void hamiltonianOMP(double hbar, vector<int>& row_ptr, vector<int>& col_idx, vector<double>& values);

double V(double r) {
    return 4.0 * eps * (pow(sigma / r, 12) - pow(sigma / r, 6));
}

void fillCPU(double* r, double dr, double rmin) {
    for (int i = 0; i < N; ++i) {
        r[i] = rmin + i * dr;
    }
}

void buildHamCPU(double dr, double hbar, double* r, vector<int>& row_ptr, vector<int>& col_idx, vector<double>& values) {
    double T = -hbar * hbar / (2.0 * 1.0 * dr * dr);

    int nnz = 0;
    row_ptr.resize(N + 1);
    for (int i = 0; i < N; i++) {
        row_ptr[i] = nnz;
        if (i > 0) nnz++;
        nnz++;
        if (i < N - 1) nnz++;
    }
    row_ptr[N] = nnz;

    col_idx.resize(nnz);
    values.resize(nnz);

    for (int i = 0; i < N; i++) {
        int offset = row_ptr[i];

        if (i > 0) {
            values[offset] = T;
            col_idx[offset] = i - 1;
            offset++;
        }

        values[offset] = -2.0 * T + V(r[i]);
        col_idx[offset] = i;
        offset++;

        if (i < N - 1) {
            values[offset] = T;
            col_idx[offset] = i + 1;
            offset++;
        }
    }
}

void hamiltonianCPU(double hbar, vector<int>& row_ptr, vector<int>& col_idx, vector<double>& values) {
    double* r = (double*)malloc(N * sizeof(double));
    fillCPU(r, dr, rmin);

    buildHamCPU(dr, hbar, r, row_ptr, col_idx, values);

    free(r);
}

__global__ void fill(double* r_host, double dr, double rmin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        r_host[i] = rmin + i * dr;
    }
}

__device__ double V_GPU(double r, double eps, double sigma) {
    return 4.0 * eps * (pow(sigma / r, 12) - pow(sigma / r, 6));
}

__global__ void buildHamCSR(double hbar, double dr,
    double eps, double sigma, double* r,
    double* H, int* col_idx, int* row_ptr, double T) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {

        int offset = row_ptr[i];

        if (i > 0) {
            H[offset] = T;
            col_idx[offset] = i - 1;
            offset++;
        }

        H[offset] = -2.0 * T + V_GPU(r[i], eps, sigma);
        col_idx[offset] = i;
        offset++;

        if (i < N - 1) {
            H[offset] = T;
            col_idx[offset] = i + 1;
            offset++;
        }
    }
}

void hamiltonianGPU(double hbar, vector<int>& h_row_ptr, vector<int>& h_col_idx, vector<double>& h_values) {
    double* d_r;
    cudaMalloc(&d_r, N * sizeof(double));

    int blocks = (N + threads - 1) / threads;
    fill << <blocks, threads >> > (d_r, dr, rmin);
    cudaDeviceSynchronize();

    int nnz = 0;
    h_row_ptr.resize(N + 1);
    for (int i = 0; i < N; i++) {
        h_row_ptr[i] = nnz;
        if (i > 0) nnz++;
        nnz++;
        if (i < N - 1) nnz++;
    }
    h_row_ptr[N] = nnz;

    h_col_idx.resize(nnz);
    h_values.resize(nnz);

    int* d_row_ptr, * d_col_idx;
    double* d_values;
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);

    double T = -hbar * hbar / (2.0 * 1.0 * dr * dr);
    buildHamCSR << <blocks, threads >> > (hbar, dr, eps, sigma, d_r, d_values, d_col_idx, d_row_ptr, T);
    cudaDeviceSynchronize();

    cudaMemcpy(h_col_idx.data(), d_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values.data(), d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_r);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
}

int main() {
    double hbar = 1.0;

    vector<int> row_ptr, col_idx;
    vector<double> values;

    hamiltonianCPU(hbar, row_ptr, col_idx, values);
    auto start_CPU = chrono::high_resolution_clock::now();
    hamiltonianCPU(hbar, row_ptr, col_idx, values);
    auto end_CPU = chrono::high_resolution_clock::now();
    double duration_CPU = chrono::duration<double, milli>(end_CPU - start_CPU).count();
    cout << "CPU: " << duration_CPU << " ms " << values[0] << endl;

    vector<int> row_ptr2, col_idx2;
    vector<double> values2;

    hamiltonianOMP(hbar, row_ptr2, col_idx2, values2);
    auto start_OMP = chrono::high_resolution_clock::now();
    hamiltonianOMP(hbar, row_ptr2, col_idx2, values2);
    auto end_OMP = chrono::high_resolution_clock::now();
    double duration_OMP = chrono::duration<double, milli>(end_OMP - start_OMP).count();
    cout << "OMP: " << duration_OMP << " ms " << values2[0] << endl;

    vector<int> row_ptr3, col_idx3;
    vector<double> values3;

    hamiltonianGPU(hbar, row_ptr3, col_idx3, values3);
    auto start_GPU = chrono::high_resolution_clock::now();
    hamiltonianGPU(hbar, row_ptr3, col_idx3, values3);
    auto end_GPU = chrono::high_resolution_clock::now();
    double duration_GPU = chrono::duration<double, milli>(end_GPU - start_GPU).count();
    cout << "GPU: " << duration_GPU << " ms " << values3[0] << endl;

    //for (int i = 0; i < 10; i++) {
    //    cout << values[3* i] << " ";s
    //}

    return 0;
}
