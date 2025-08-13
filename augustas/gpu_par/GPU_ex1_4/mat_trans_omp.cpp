#include <omp.h>

void matTransCPU(float * A, float * B, int N, int M) {
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			B[j * N + i] = A[i * M + j];
		}
	}
}