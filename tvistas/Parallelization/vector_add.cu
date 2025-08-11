#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(float* a, float* b, float* result, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\nCompute Capability: %d.%d\nCores: %d\nMax Threads/Block: %d\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount * 128, // Approximate CUDA cores (SMs × 128)
           prop.maxThreadsPerBlock);

    int N = 2048;
    size_t size = N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_result = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1;
        h_b[i] = 2;
    }

    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // Copy result back to host
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    // Print arrays
    printf("A: ");
    for (int i = 0; i < N; i++) printf("%f, ", h_a[i]);
    printf("\nB: ");
    for (int i = 0; i < N; i++) printf("%f, ", h_b[i]);
    printf("\nResult: ");
    for (int i = 0; i < N; i++) printf("%f, ", h_result[i]);
    printf("\n");

    printf("Kernel execution time: %.6f ms\n", milliseconds);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

