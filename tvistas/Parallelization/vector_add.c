#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  const int N = 2048;
  const int numThreads = 4;
  omp_set_num_threads(numThreads);

  double start_total = omp_get_wtime();

  float *a = (float *)malloc(N * sizeof(float));
  float *b = (float *)malloc(N * sizeof(float));
  float *result = (float *)malloc(N * sizeof(float));

  double start_init = omp_get_wtime();
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }
  double end_init = omp_get_wtime();

  double start_add = omp_get_wtime();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    result[i] = a[i] + b[i];
  }
  double end_add = omp_get_wtime();

  for (int i = 0; i < N; i++) {
    printf("%f, ", result[i]);
  }

  free(a);
  free(b);
  free(result);

  double end_total = omp_get_wtime();

  printf("\n\nTiming results:\n");
  printf("Initialization time: %f seconds\n", end_init - start_init);
  printf("Addition time: %f seconds\n", end_add - start_add);
  printf("Total execution time: %f seconds\n", end_total - start_total);

  return 0;
}
