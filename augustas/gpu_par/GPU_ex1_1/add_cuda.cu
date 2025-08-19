#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void Addition(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vectorAddGPU(int* h_a, int* h_b, int* h_c, int n, int threads) {
    int* d_a;
    int* d_b;
    int* d_c;

    size_t size = n * sizeof(int);

    auto start_alloc = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    auto end_alloc = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = threads;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();
    Addition << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto start_transf = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    auto end_transf = std::chrono::high_resolution_clock::now();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    double allocation = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
    double transfer = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();
    double execution = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "GPU allocation time: " << allocation + transfer << " ms\n";
    std::cout << "GPU processing time: " << execution << " ms\n";
}

void GPU_properties() {

    cudaDeviceProp prop;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads in each dimension: "
        << "x=" << prop.maxThreadsDim[0] << ", "
        << "y=" << prop.maxThreadsDim[1] << ", "
        << "z=" << prop.maxThreadsDim[2] << std::endl;

}