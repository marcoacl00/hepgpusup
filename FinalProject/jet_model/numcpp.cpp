// Header file
#include "numcpp.hpp"

namespace numcpp
{
	vector_t linspace(const float start, const float stop, const unsigned long NUM) noexcept(false)
	{
		if (start >= stop)
			throw "Invalid range";

		if (NUM <= 0)
			throw "Invalid number of points";

		const float STEP = (stop - start) / NUM;
		vector_t arr(NUM);
		arr[0] = start;

		for (unsigned long i = 1; i < NUM; ++i)
			arr[i] = arr[i - 1] + STEP;

		return arr;
	}

	void meshgrid(const vector_t& x, const vector_t& y, matrix_t& X, matrix_t& Y) noexcept(true)
	{
		const unsigned long
			X_DIM = x.size(),
			Y_DIM = y.size();

		for (unsigned long i = 0; i < X_DIM; ++i)
		{
			for (unsigned long j = 0; j < Y_DIM; ++j)
			{
				X[i][j] = x[i];
				Y[i][j] = y[j];
			}
		}
	}

	matrix_t roll(const matrix_t& mtx, unsigned long offset, const axis_t axis) noexcept(true)
	{
		const int
			X_DIM = mtx.size(),
			Y_DIM = mtx[0].size();

		int index;
		matrix_t ret(X_DIM, vector_t(Y_DIM, 0));

		if (axis == axis_t::X)
		{
			offset %= X_DIM;

			for (int i = 0; i < X_DIM; ++i)
			{
				for (int j = 0; j < Y_DIM; ++j)
				{
					index = i - offset;

					if (index < 0)
						index += X_DIM;
					else if (index >= X_DIM)
						index -= X_DIM;
					
					ret[i][j] = mtx[index][j];
				}
			}
		}
		else
		{
			offset %= Y_DIM;

			for (int i = 0; i < X_DIM; ++i)
			{
				for (int j = 0; j < Y_DIM; ++j)
				{
					index = j - offset;

					if (index < 0)
						index += Y_DIM;
					else if (index >= Y_DIM)
						index -= Y_DIM;
					
					ret[i][j] = mtx[i][index];
				}
			}
		}

		return ret;
	}
}

vector_t operator - (const vector_t& vec_1, const vector_t& vec_2) noexcept(true)
{
	const unsigned long DIM = vec_1.size();
	vector_t ret(DIM);

	for (unsigned long i = 0; i < DIM; ++i)
		ret[i] = vec_1[i] - vec_2[i];

	return ret;
}

matrix_t operator - (const matrix_t& mtx_1, const matrix_t& mtx_2) noexcept(true)
{
	const unsigned long
		X_DIM = mtx_1.size(),
		Y_DIM = mtx_1[0].size();
	matrix_t ret(X_DIM, vector_t(Y_DIM));

	for (unsigned long i = 0; i < X_DIM; ++i)
	{
		for (unsigned long j = 0; j < Y_DIM; ++j)
			ret[i][j] = mtx_1[i][j] - mtx_2[i][j];
	}

	return ret;
}

matrix_t operator * (const float scalar, const matrix_t& mtx) noexcept(true)
{
	const unsigned long
		X_DIM = mtx.size(),
		Y_DIM = mtx[0].size();
	matrix_t ret(X_DIM, vector_t(Y_DIM));

	for (unsigned long i = 0; i < X_DIM; ++i)
	{
		for (unsigned long j = 0; j < Y_DIM; ++j)
			ret[i][j] = mtx[i][j] * scalar;
	}

	return ret;
}

matrix_t operator / (const matrix_t& mtx, const float scalar) noexcept(true)
{
	const unsigned long
		X_DIM = mtx.size(),
		Y_DIM = mtx[0].size();
	matrix_t ret(X_DIM, vector_t(Y_DIM));

	for (unsigned long i = 0; i < X_DIM; ++i)
	{
		for (unsigned long j = 0; j < Y_DIM; ++j)
			ret[i][j] = mtx[i][j] / scalar;
	}

	return ret;
}

matrix_t operator + (const matrix_t& mtx_1, const matrix_t& mtx_2) noexcept(true)
{
	const unsigned long
		X_DIM = mtx_1.size(),
		Y_DIM = mtx_1[0].size();
	matrix_t ret(X_DIM, vector_t(Y_DIM));

	for (unsigned long i = 0; i < X_DIM; ++i)
	{
		for (unsigned long j = 0; j < Y_DIM; ++j)
			ret[i][j] = mtx_1[i][j] + mtx_2[i][j];
	}

	return ret;
}

matrix_t operator * (const matrix_t& mtx_1, const matrix_t& mtx_2) noexcept(true)
{
	const unsigned long
		X_DIM = mtx_1.size(),
		Y_DIM = mtx_1[0].size();
	matrix_t ret(X_DIM, vector_t(Y_DIM));

	for (unsigned long i = 0; i < X_DIM; ++i)
	{
		for (unsigned long j = 0; j < Y_DIM; ++j)
			ret[i][j] = mtx_1[i][j] * mtx_2[i][j];
	}

	return ret;
}
