#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int nThreads = 4;
  int N = 10000000;

  double start = omp_get_wtime();

  float *a = (float *)malloc(N * sizeof(float));
  float *result = (float *)malloc(N * sizeof(float));

  double finishAlloc = omp_get_wtime();

#pragma omp parallel for
  for (int i = 0; i < N; i++)
    a[i] = (float)i / N;

  double finishInit = omp_get_wtime();

#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    result[i] = sinf(cosf(logf(a[i])));
  }

  double finishExecution = omp_get_wtime();

  for (int i = 0; i < 10; ++i) {
    printf("f(%f) = %f\n", a[i], result[i]);
  }

  printf("Time to allocate memory: %f\n", finishAlloc - start);
  printf("Time to initialize memory: %f\n", finishInit - finishAlloc);
  printf("Time to execute: %f\n", finishExecution - finishInit);

  free(a);
  free(result);
  return 0;
}
