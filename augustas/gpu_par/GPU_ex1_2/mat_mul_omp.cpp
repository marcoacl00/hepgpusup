#include <omp.h>

void matrixMulCPU(int* A, int* B, float* C, int N, int M, int K) {
#pragma omp parallel for
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float sum = 0.0;
			for (int k = 0; k < K; ++k) {
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}