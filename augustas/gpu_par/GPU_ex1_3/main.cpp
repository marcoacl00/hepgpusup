#include <iostream>
#include <chrono>

using namespace std;

void matVecMulCPU(float* A, float* b, float* c, int N, int M);
void matVecMulGPU(float* h_A, float* h_b, float* h_c, int N, int M, int threads);

void fillMatrix(float* A, int row, int col) {
	for (int i = 0; i < row * col; i++) {
		A[i] = i;
	}
}

void fillVector(float* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}
}

void testing(int N, int M, int threads_gpu) {
	auto start_cpu_alloc = std::chrono::high_resolution_clock::now();
	float* A = (float*)malloc(M * N * sizeof(float));
	float* b = (float*)malloc(M * sizeof(float));
	float* c_cpu = (float*)malloc(N * sizeof(float));
	auto end_cpu_alloc = std::chrono::high_resolution_clock::now();
	float* c_gpu = (float*)malloc(N * sizeof(float));

	fillMatrix(A, N, M);
	fillVector(b, M);

	auto start_cpu = chrono::high_resolution_clock::now();
	matVecMulCPU(A, b, c_cpu, N, M);
	auto end_cpu = chrono::high_resolution_clock::now();

	cout << "CPU allocation time: " << chrono::duration<double, std::milli>(end_cpu_alloc - start_cpu_alloc).count() << " ms\n";
	cout << "CPU processing time: " << chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms\n";

	matVecMulGPU(A, b, c_gpu, N, M, threads_gpu);

	int errors = 0;
	for (int i = 0; i < N; ++i) {
		if (fabs(c_cpu[i] - c_gpu[i]) > 1e-5) {
			errors++;
			std::cout << "Error at index " << i << ": CPU=" << c_cpu[i] << ", GPU=" << c_gpu[i] << "\n";
		}
	}
	//cout << c_cpu[N - 2] << " " << c_gpu[N - 2] << endl;

	std::cout << "Results mismatch: " << errors << endl;

	free(A);
	free(b);
	free(c_cpu);
	free(c_gpu);
}

int main() {
	testing(128, 128, 32);
	int N = 1024, M = 128;
	int threads = 32;
	for (int i = 0; i < 7; i++) {
		testing(N, M, threads);
		M *= 2;
		N *= 2;
	}
}