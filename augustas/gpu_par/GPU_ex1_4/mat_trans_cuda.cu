#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void Transpose(float *A, float * B, int N, int M) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M) {
		B[j * N + i] = A[i * M + j];
	}
}

void matTransGPU(float* h_A, float* h_B, int N, int M, int threads) {
	float* d_A, * d_B;

	auto start_alloc = std::chrono::high_resolution_clock::now();
	cudaMalloc(&d_A, N * M * sizeof(float));
	cudaMalloc(&d_B, N * M * sizeof(float));

	cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	auto end_alloc = std::chrono::high_resolution_clock::now();

	dim3 blockDim(threads, threads);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
		(M + blockDim.y - 1) / blockDim.y);

	auto start = std::chrono::high_resolution_clock::now();
	Transpose << <gridDim, blockDim >> > (d_A, d_B, N, M);
	auto end = std::chrono::high_resolution_clock::now();

	auto start_transf = std::chrono::high_resolution_clock::now();
	cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	auto end_transf = std::chrono::high_resolution_clock::now();

	cudaFree(d_A);
	cudaFree(d_B);

	double allocation = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
	double transfer = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
	double execution = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "GPU allocation time: " << allocation + transfer << " ms\n";
	std::cout << "GPU processing time: " << execution << " ms\n";
}