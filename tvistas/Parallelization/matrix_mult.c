

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
  int nThreads = 4;
  int M = 1024;
  int N = 2048;
  int K = 512;

  double start = omp_get_wtime();

  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *Result = (float *)malloc(M * N * sizeof(float));

  double finishAlloc = omp_get_wtime();

#pragma omp parallel for
  for (int i = 0; i < M * K; i++)
    A[i] = 1.0f;
#pragma omp parallel for
  for (int i = 0; i < K * N; i++)
    B[i] = 2.0f;

  double finishInit = omp_get_wtime();

#pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        Result[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  double finishExecution = omp_get_wtime();

  printf("Result matrix C (%d x %d):\n", M, N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%6.1f ", Result[i * N + j]);
    }
    printf("\n");
  }
  
  printf("Time to allocate memory: %f\n", finishAlloc - start);
  printf("Time to initialize memory: %f\n", finishInit - finishAlloc);
  printf("Time to execute: %f\n", finishExecution - finishInit);

  free(A);
  free(B);
  free(Result);
  return 0;
}
