#include <vector>

const float sigma = 2.099518748076819;
const float eps = 4.642973059984742;
const float mu = 1.0;
const int N = 15000000;
const float rmin = 2.0;
const float rmax = 2.7;
const float dr = (rmax - rmin) / (N);

float V(float r);

void fillOMP(float* r, float dr, float rmin) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        r[i] = rmin + i * dr;
    }
}

void buildHamOMP(float dr, float hbar, float* r,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    std::vector<float>& values)
{
    float T = -hbar * hbar / (2.0 * 1.0 * dr * dr);

    // Step 1: row_ptr
    int nnz = 0;
    row_ptr.resize(N + 1);
    for (int i = 0; i < N; i++) {
        row_ptr[i] = nnz;
        if (i > 0) nnz++;
        nnz++; // diagonal
        if (i < N - 1) nnz++;
    }
    row_ptr[N] = nnz;

    col_idx.resize(nnz);
    values.resize(nnz);

    // Step 2: fill CSR
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        int offset = row_ptr[i];

        if (i > 0) {
            values[offset] = T;
            col_idx[offset] = i - 1;
            offset++;
        }

        values[offset] = -2.0 * T + V(r[i]);
        col_idx[offset] = i;
        offset++;

        if (i < N - 1) {
            values[offset] = T;
            col_idx[offset] = i + 1;
            offset++;
        }
    }
}

void hamiltonianOMP(float hbar,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    std::vector<float>& values)
{
    float* r = (float*)malloc(N * sizeof(float));
    fillOMP(r, dr, rmin);

    buildHamOMP(dr, hbar, r, row_ptr, col_idx, values);

    free(r);
}
