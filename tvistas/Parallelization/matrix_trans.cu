#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void transposeKernel(const float* A, float* A_T, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        A_T[col * M + row] = A[row * N + col];
    }
}

int main() {
    int M = 40;
    int N = 3; 

    size_t sizeA = M * N * sizeof(float);
    size_t sizeAT = N * M * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_AT = (float*)malloc(sizeAT);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = i * N + j + 1;
        }
    }

    float *d_A, *d_AT;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_AT, sizeAT);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    transposeKernel<<<gridSize, blockSize>>>(d_A, d_AT, M, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Transpose kernel execution time: %.6f ms\n", ms);

    cudaMemcpy(h_AT, d_AT, sizeAT, cudaMemcpyDeviceToHost);

    printf("Original matrix A (%d x %d):\n", M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4.0f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    printf("Transposed matrix A_T (%d x %d):\n", N, M);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%4.0f ", h_AT[i * M + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_AT);
    free(h_A);
    free(h_AT);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

