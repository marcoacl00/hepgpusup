#include <chrono>
#include <iostream>

using namespace std;

void matTransGPU(float* h_A, float* h_B, int N, int M, int threads);
void matTransCPU(float* A, float* B, int N, int M);

void fillMatrix(float* A, int row, int col) {
	for (int i = 0; i < row * col; i++) {
		A[i] = i;
	}
}

void testing(int N, int M, int threads_gpu) {
	auto start_cpu_alloc = std::chrono::high_resolution_clock::now();
	float* A = (float*)malloc(N * M * sizeof(float));
	float* B_cpu = (float*)malloc(N *M * sizeof(float));
	auto end_cpu_alloc = std::chrono::high_resolution_clock::now();
	float* B_gpu = (float*)malloc(N * M * sizeof(float));

	fillMatrix(A, N, M);

	auto start_cpu = std::chrono::high_resolution_clock::now();
	matTransCPU(A, B_cpu, N, M);
	auto end_cpu = std::chrono::high_resolution_clock::now();

	cout << "CPU allocation time: " << std::chrono::duration<double, std::milli>(end_cpu_alloc - start_cpu_alloc).count() << " ms\n";
	cout << "CPU processing time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms\n";

	matTransGPU(A, B_gpu, N, M, threads_gpu);

	int errors = 0;
	for (int i = 0; i < M * N; i++) {
		/*cout << B_gpu[i] << " ";
		if ((i+1) % N == 0) {
			cout << endl;
		}*/
		if (fabs(B_cpu[i] - B_gpu[i]) > 1e-5) {
			errors++;
			std::cout << "Error at index " << i << ": CPU=" << B_cpu[i] << ", GPU=" << B_gpu[i] << "\n";
		}
	}

	std::cout << "Results mismatch: " << errors << endl;

	free(A);
	free(B_cpu);
	free(B_gpu);
}

int main() {
	testing(4, 4, 1);
	int N = 256, M = 256;
	int threads = 8;
	for (int i = 0; i < 8; i++) {
		testing(N, M, threads);
		M *= 2;
		N *= 2;
	}
}