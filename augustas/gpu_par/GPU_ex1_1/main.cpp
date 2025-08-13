#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;

void vectorAddCPU(int* a, int* b, int* c, int n);
void vectorAddGPU(int* a, int* b, int* c, int n, int threads);
void GPU_properties();


void testing(int n, int threads_gpu) {
    auto start_cpu_alloc = std::chrono::high_resolution_clock::now();
    int* a = (int*)malloc(n * sizeof(int));
    int* b = (int*)malloc(n * sizeof(int));
    int* c_cpu = (int*)malloc(n * sizeof(int));
    auto end_cpu_alloc = std::chrono::high_resolution_clock::now();
    int* c_gpu = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(a, b, c_cpu, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    cout << "CPU allocation time: " << std::chrono::duration<double, std::milli>(end_cpu_alloc - start_cpu_alloc).count() << " ms\n";
    cout << "CPU processing time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms\n";

    vectorAddGPU(a, b, c_gpu, n, threads_gpu);

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (fabs(c_cpu[i] - c_gpu[i]) > 1e-5) {
            errors++;
            if (errors < 10) {
                std::cout << "Error at index " << i << ": CPU=" << c_cpu[i] << ", GPU=" << c_gpu[i] << "\n";
            }
        }
    }

    std::cout << "Results mismatch: " << errors << endl;

    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
}

int main() {
    // initial test for more accurate testing
    testing(100000000, 256);
    /*int n = 50000;
    int threads = 32;
    for (int i = 0; i < 5; i++) {
        //testing(n, threads);
        n *= 10;
    }*/

    //GPU_properties();
}
