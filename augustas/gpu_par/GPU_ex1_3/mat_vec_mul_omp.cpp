#include <omp.h>

void matVecMulCPU(float* A, float* b, float* c, int N, int M) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            sum += A[i * M + j] * b[j];
        }
        c[i] = sum;
    }
}