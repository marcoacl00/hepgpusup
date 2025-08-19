#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void Multiplication(float* A, float* b, float* c, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0;
        for (int j = 0; j < M; j++) {
            sum += A[i * M + j] * b[j];
        }
        c[i] = sum;
    }
}

void matVecMulGPU(float* h_A, float* h_b, float* h_c, int N, int M, int threads) {
    float* d_A, * d_b, * d_c;

    auto start_alloc = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_b, M * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, M * sizeof(float), cudaMemcpyHostToDevice);
    auto end_alloc = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = threads;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    auto start = std::chrono::high_resolution_clock::now();
    Multiplication << <blocksPerGrid, threadsPerBlock >> > (d_A, d_b, d_c, N, M);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto start_transf = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    auto end_transf = std::chrono::high_resolution_clock::now();

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_c);

    double allocation = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
    double transfer = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
    double execution = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "GPU allocation time: " << allocation + transfer << " ms\n";
    std::cout << "GPU processing time: " << execution << " ms\n";
}