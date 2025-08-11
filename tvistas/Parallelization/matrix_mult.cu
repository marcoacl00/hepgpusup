#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiplication(float* A, float* B, float* Result, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M (rows of A, C)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N (columns of B, C)

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        Result[row * N + col] = sum;
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
    int M  = 1024;
    int K = 512;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_result = (float*)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_result;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_result, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); // Threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_result, M, N, K);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix multiplication kernel time: %.6f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, sizeC, cudaMemcpyDeviceToHost);

    // Print result (for small matrices)
    printf("Result matrix C (%d x %d):\n", M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%6.1f ", h_result[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    free(h_A);
    free(h_B);
    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;    
}

