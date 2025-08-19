#include <cuda_runtime.h>
#include <cusparse.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

const float sigma = 2.099518748076819;
const float eps = 4.642973059984742;
const float mu = 1.0;
const int N = 10000000;
const int threads = 1024;
const float rmin = 2.0;
const float rmax = 2.7;
const float dr = (rmax - rmin) / (N);


void hamiltonianOMP(float hbar, vector<int>& row_ptr, vector<int>& col_idx, vector<float>& values);

float V(float r) {
    return 4.0 * eps * (pow(sigma / r, 12) - pow(sigma / r, 6));
}

void fillCPU(float* r, float dr, float rmin) {
    for (int i = 0; i < N; ++i) {
        r[i] = rmin + i * dr;
    }
}

void buildHamCPU(float dr, float hbar, float* r, vector<int>& row_ptr, vector<int>& col_idx, vector<float>& values) {
    float T = -hbar * hbar / (2.0 * 1.0 * dr * dr);

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

void hamiltonianCPU(float hbar, vector<int>& row_ptr, vector<int>& col_idx, vector<float>& values) {
    float* r = (float*)malloc(N * sizeof(float));
    fillCPU(r, dr, rmin);

    buildHamCPU(dr, hbar, r, row_ptr, col_idx, values);

    free(r);
}

__global__ void fill(float* r_host, float dr, float rmin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        r_host[i] = rmin + i * dr;
    }
}

__device__ float V_GPU(float r, float eps, float sigma) {
    return 4.0 * eps * (pow(sigma / r, 12) - pow(sigma / r, 6));
}

__global__ void buildHamCSR(float hbar, float dr,
    float eps, float sigma, float* r,
    float* H, int* col_idx, int* row_ptr, float T) {

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

void hamiltonianGPU(float hbar, vector<int>& h_row_ptr, vector<int>& h_col_idx, vector<float>& h_values) {
    float* d_r;
    cudaMalloc(&d_r, N * sizeof(float));

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
    float* d_values;
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);

    float T = -hbar * hbar / (2.0 * 1.0 * dr * dr);
    buildHamCSR << <blocks, threads >> > (hbar, dr, eps, sigma, d_r, d_values, d_col_idx, d_row_ptr, T);
    cudaDeviceSynchronize();

    cudaMemcpy(h_col_idx.data(), d_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values.data(), d_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_r);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
}

int main() {
    float hbar = 1.0;

    vector<int> row_ptr, col_idx;
    vector<float> values;

    hamiltonianCPU(hbar, row_ptr, col_idx, values);
    auto start_CPU = chrono::high_resolution_clock::now();
    hamiltonianCPU(hbar, row_ptr, col_idx, values);
    auto end_CPU = chrono::high_resolution_clock::now();
    float duration_CPU = chrono::duration<float, milli>(end_CPU - start_CPU).count();
    cout << "CPU: " << duration_CPU << " ms " << values[0] << endl;

    vector<int> row_ptr2, col_idx2;
    vector<float> values2;

    hamiltonianOMP(hbar, row_ptr2, col_idx2, values2);
    auto start_OMP = chrono::high_resolution_clock::now();
    hamiltonianOMP(hbar, row_ptr2, col_idx2, values2);
    auto end_OMP = chrono::high_resolution_clock::now();
    float duration_OMP = chrono::duration<float, milli>(end_OMP - start_OMP).count();
    cout << "OMP: " << duration_OMP << " ms " << values2[0] << endl;

    vector<int> row_ptr3, col_idx3;
    vector<float> values3;

    hamiltonianGPU(hbar, row_ptr3, col_idx3, values3);
    auto start_GPU = chrono::high_resolution_clock::now();
    hamiltonianGPU(hbar, row_ptr3, col_idx3, values3);
    auto end_GPU = chrono::high_resolution_clock::now();
    float duration_GPU = chrono::duration<float, milli>(end_GPU - start_GPU).count();
    cout << "GPU: " << duration_GPU << " ms " << values3[0] << endl;

    //for (int i = 0; i < 10; i++) {
    //    cout << values[3* i] << " ";s
    //}

    return 0;
}
