#include <omp.h>
#include <cmath>

void funcEvalCPU(float * x, float* y, int n) {
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		y[i] = sin(cos(log(x[i])));
	}
}