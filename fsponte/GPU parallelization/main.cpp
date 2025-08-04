#include <iostream>
#include <cmath>

#include "vector.hpp"
#include "matrix.hpp"

template <typename type_t, unsigned long DIM>
vector<type_t, DIM> fn(vector<type_t, DIM>);

int main()
{
	vector<double, 3> vec;
	matrix<double, 3, 3> mtx;

	// Setup vector
	{
		vec[0] = 1;
		vec[1] = 3;
		vec[2] = 9;
		vec.print();
	}

	// Setup matrix
	{
		mtx[0][0] = 1;
		mtx[0][1] = 2;
		mtx[0][2] = 3;
		mtx[1][0] = 4;
		mtx[1][1] = 5;
		mtx[1][2] = 6;
		mtx[2][0] = 7;
		mtx[2][1] = 8;
		mtx[2][2] = 9;
		mtx.print();

		mtx = mtx.transpose();
		mtx.print();
	}

	vector<double, 3> res = mtx * vec;
	res.print();
	res = fn(res);
	res.print();

	return 0;
}

template <typename type_t, unsigned long DIM>
vector<type_t, DIM> fn(vector<type_t, DIM> vec)
{
	vector<type_t, DIM> ret;

	for (unsigned long i = 0; i < DIM; ++i)
		ret[i] = std::sin(std::cos(std::log(vec[i])));

	return ret;
}
