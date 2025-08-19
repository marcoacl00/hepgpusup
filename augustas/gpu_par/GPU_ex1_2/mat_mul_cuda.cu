#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void Multiplication(int* A, int* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0.0f;

    if (row < M && col < N) {
        for (int i = 0; i < K; i++) {
            tmpSum += A[row * K + i] * B[i * N + col];
        }
    }
    C[row * N + col] = tmpSum;
}

void matrixMulGPU(int* h_A, int* h_B, float* h_C, int M, int N, int K, int threads) {
    int* d_A, * d_B;
    float *d_C;

    auto start_alloc = std::chrono::high_resolution_clock::now();
	cudaMalloc(&d_A, M * K * sizeof(int));
	cudaMalloc(&d_B, K * N * sizeof(int));
	cudaMalloc(&d_C, M * N * sizeof(float));

	cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K * N * sizeof(int), cudaMemcpyHostToDevice);
    auto end_alloc = std::chrono::high_resolution_clock::now();

    dim3 threadsPerBlock(threads,threads);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto start = std::chrono::high_resolution_clock::now();
    Multiplication << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto start_transf = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    auto end_transf = std::chrono::high_resolution_clock::now();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    double allocation = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
    double transfer = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
    double execution = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "GPU allocation time: " << allocation + transfer << " ms\n";
    std::cout << "GPU processing time: " << execution << " ms\n";
}