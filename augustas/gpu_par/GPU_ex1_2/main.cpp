#include <iostream>
#include <chrono>

using namespace std;

void matrixMulGPU(int* h_A, int* h_B, float* h_C, int M, int N, int K, int threads);
void matrixMulCPU(int* A, int* B, float* C, int N, int M, int K);

void fillMatrix(int* A, int row, int col) {
	for (int i = 0; i < row * col; i++) {
		A[i] = i;
	}
}

void testing(int M, int N, int K, int threads_gpu) {
	auto start_cpu_alloc = std::chrono::high_resolution_clock::now();
	int* A = (int*)malloc(M * K * sizeof(int));
	int* B = (int*)malloc(K * N * sizeof(int));
	float* C_cpu = (float*)malloc(M * N * sizeof(float));
	auto end_cpu_alloc = std::chrono::high_resolution_clock::now();
	float* C_gpu = (float*)malloc(M * N * sizeof(float));

	fillMatrix(A, M, K);
	fillMatrix(B, K, N);

	auto start_cpu = std::chrono::high_resolution_clock::now();
	matrixMulCPU(A, B, C_cpu, N, M, K);
	auto end_cpu = std::chrono::high_resolution_clock::now();

	cout << "CPU allocation time: " << std::chrono::duration<double, std::milli>(end_cpu_alloc - start_cpu_alloc).count() << " ms\n";
	cout << "CPU processing time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms\n";

	matrixMulGPU(A, B, C_gpu, M, N, K, threads_gpu);

	int errors = 0;
	for (int i = 0; i < M * N; ++i) {
		if (fabs(C_cpu[i] - C_gpu[i]) > 1e-5) {
			errors++;
			std::cout << "Error at index " << i << ": CPU=" << C_cpu[i] << ", GPU=" << C_gpu[i] << "\n";
		}
	}

	std::cout << "Results mismatch: " << errors << endl;

	free(A);
	free(B);
	free(C_cpu);
	free(C_gpu);
}

int main() {
	testing(128, 128, 128, 32);
	int M = 32, N = 32, K = 32;
	int threads = 32;
	for (int i = 0; i < 8; i++) {
		testing(M, N, K, threads);
		M *= 2;
		N *= 2;
		K *= 2;
	}
}