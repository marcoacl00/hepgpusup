#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int nThreads = 4;
  int M = 1024;
  int N = 1024;

  double start = omp_get_wtime();

  float *A = (float *)malloc(M * N * sizeof(float));
  float *AT = (float *)malloc(M * N * sizeof(float));

  double finishAlloc = omp_get_wtime();

#pragma omp parallel for
  for (int i = 0; i < M * N; i++)
    A[i] = i;

  double finishInit = omp_get_wtime();

#pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        AT[j * M + i] = A[i * N + j];
    }
  }

  double finishExecution = omp_get_wtime();

  printf("Original matrix (%d x %d):\n", M, N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%6.1f ", A[i * N + j]);
    }
    printf("\n");
  }

  printf("Result matrix (%d x %d):\n", N, M);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      printf("%6.1f ", AT[i * M + j]);
    }
    printf("\n");
  }

  printf("Time to allocate memory: %f\n", finishAlloc - start);
  printf("Time to initialize memory: %f\n", finishInit - finishAlloc);
  printf("Time to execute: %f\n", finishExecution - finishInit);

  free(A);
  free(AT);
  return 0;
}
