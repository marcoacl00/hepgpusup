#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void Evaluate(float* x, float *y, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] = sin(cos(log(x[i])));
	}
}

void funcEvalGPU(float* h_x, float * h_y, int n, int threads) {
	float* d_x, * d_y;

	auto start_alloc = std::chrono::high_resolution_clock::now();
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));

	cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
	auto end_alloc = std::chrono::high_resolution_clock::now();

	int threadsPerBlock = threads;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	auto start = std::chrono::high_resolution_clock::now();
	Evaluate << <blocksPerGrid, threadsPerBlock >> > (d_x, d_y, n);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	auto start_transf = std::chrono::high_resolution_clock::now();
	cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
	auto end_transf = std::chrono::high_resolution_clock::now();

	cudaFree(d_x);
	cudaFree(d_y);

	double allocation = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
	double transfer = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
	double execution = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "GPU allocation time: " << allocation + transfer << " ms\n";
	std::cout << "GPU processing time: " << execution << " ms\n";
}