#include <omp.h>
#include <fstream>

void vectorAddCPU(int* a, int* b, int* c, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
        //printf("%d\n", omp_get_thread_num());
    }
}
