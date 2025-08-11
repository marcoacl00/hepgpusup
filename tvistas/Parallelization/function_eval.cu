#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void f(const float* A, float* result, int N) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i< N) {
        result[i] = sinf(cosf(logf(A[i])));
    }
}

int main() {
    int N = 10000000; 

    size_t size = N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_result = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_a[i] = (i + 0.5)/N;
    }

    float *d_a, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    f<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_result, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("f(%f) = %f\n", h_a[i], h_result[i]);
    }

    printf("Function ( sin(cos(log(x))) ) kernel execution time: %.6f ms\n", ms);
    
    cudaFree(d_a);
    cudaFree(d_result);
    free(h_a);
    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

