#include <iostream>

#include "vector.hpp"
#include "matrix.hpp"

template <typename type_t, unsigned long DIM>
void print(const vector<type_t, DIM>&);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void print(const matrix<type_t, N_LIN, N_COL>&);

int main()
{
	matrix<double, 3, 3> mtx_1, mtx_2, mtx_3;

	mtx_1[0][0] = 1;
	mtx_1[0][1] = 2;
	mtx_1[0][2] = 3;
	mtx_1[1][0] = 4;
	mtx_1[1][1] = 5;
	mtx_1[1][2] = 6;
	mtx_1[2][0] = 7;
	mtx_1[2][1] = 8;
	mtx_1[2][2] = 9;

	mtx_2[0][0] = 9;
	mtx_2[0][1] = 8;
	mtx_2[0][2] = 7;
	mtx_2[1][0] = 6;
	mtx_2[1][1] = 5;
	mtx_2[1][2] = 4;
	mtx_2[2][0] = 3;
	mtx_2[2][1] = 2;
	mtx_2[2][2] = 1;

	mtx_3 = mtx_1 + mtx_2;

	print(mtx_1);
	print(mtx_2);
	print(mtx_3);

	return 0;
}

template <typename type_t, unsigned long DIM>
void print(const vector<type_t, DIM>& vec)
{
	std::cout << '(' << vec[0];

	for (unsigned long i = 1; i < DIM; ++i)
		std::cout << ", " << vec[i];

	std::cout << ")\n";
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void print(const matrix<type_t, N_LIN, N_COL>& mtx)
{
	std::cout << "{\n";

	for (unsigned long i = 0; i < N_LIN; ++i)
		print(mtx[i]);

	std::cout << "}\n";
}
