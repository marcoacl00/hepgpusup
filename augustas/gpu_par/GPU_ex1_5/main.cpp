#include <chrono>
#include <iostream>

using namespace std;

void funcEvalCPU(float* x, float* y, int n);
void funcEvalGPU(float* h_x, float* h_y, int n, int threads);

void testing(int n, int threads_gpu) {
	auto start_cpu_alloc = std::chrono::high_resolution_clock::now();
	float* x = (float*)malloc(n * sizeof(float));
	float* y_cpu = (float*)malloc(n * sizeof(float));
	auto end_cpu_alloc = std::chrono::high_resolution_clock::now();
	float* y_gpu = (float*)malloc(n * sizeof(float));

	for (int i = 0; i < n; i++) {
		x[i] = 1.0 + i * 0.001;
	}

	auto start_cpu = std::chrono::high_resolution_clock::now();
	funcEvalCPU(x, y_cpu, n);
	auto end_cpu = std::chrono::high_resolution_clock::now();

	cout << "CPU allocation time: " << std::chrono::duration<double, std::milli>(end_cpu_alloc - start_cpu_alloc).count() << " ms\n";
	cout << "CPU processing time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms\n";

	funcEvalGPU(x, y_gpu, n, threads_gpu);

	int errors = 0;
	for (int i = 0; i < n; i++) {
		//cout << x[i] << " " << y_cpu[i] << "\n";
		if (fabs(y_cpu[i] - y_gpu[i]) > 1e-5) {
			errors++;
			std::cout << "Error at index " << i << ": CPU=" << y_cpu[i] << ", GPU=" << y_gpu[i] << "\n";
		}
	}

	std::cout << "Results mismatch: " << errors << endl;

	free(x);
	free(y_cpu);
	free(y_gpu);
}

int main() {
	testing(1000, 32);
	int n = 10000000;
	int threads = 256;
	for (int i = 0; i < 5; i++) {
		testing(n, threads);
		n *= 2;
	}
}