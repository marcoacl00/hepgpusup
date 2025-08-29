#ifndef _NUMCPP_TPP_
#define _NUMCPP_TPP_

namespace numcpp
{
	template <typename type_t, unsigned long DIM>
	vector_t<type_t, DIM> linspace(const type_t start, const type_t stop) noexcept(false)
	{
		if (start >= stop || start >= DIM || stop >= DIM)
			throw "Invalid range";

		const type_t STEP = (stop - start) / DIM;
		vector_t<type_t, DIM> arr;
		arr[0] = start;

		for (unsigned long i = 1; i < DIM; ++i)
			arr[i] = arr[i - 1] + STEP;

		return arr;
	}

	template <typename type_t, unsigned long DIM>
	void meshgrid(const vector_t<type_t, DIM>& x, const vector_t<type_t, DIM>& y, matrix_t<type_t, DIM, DIM>& X, matrix_t<type_t, DIM, DIM>& Y) noexcept(true)
	{
		for (unsigned long i = 0; i < DIM; ++i)
		{
			for (unsigned long j = 0; j < DIM; ++j)
			{
				X.get(i, j) = x[i];
				Y.get(i, j) = y[j];
			}
		}
	}

	template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
	matrix_t<type_t, N_LIN, N_COL> roll(const matrix_t<type_t, N_LIN, N_COL>& mtx, unsigned long offset, const axis_t axis) noexcept(true)
	{
		unsigned long index;
		matrix_t<type_t, N_LIN, N_COL> ret;

		if (axis == axis_t::X)
		{
			offset %= N_LIN;

			for (unsigned long i = 0; i < N_LIN; ++i)
			{
				for (unsigned long j = 0; j < N_COL; ++j)
				{
					index = (i + N_LIN - offset) % N_LIN;
					ret.get(i, j) = mtx.get(index, j);
				}
			}
		}
		else
		{
			offset %= N_COL;

			for (unsigned long i = 0; i < N_LIN; ++i)
			{
				for (unsigned long j = 0; j < N_COL; ++j)
				{
					index = (j + N_COL - offset) % N_COL;
					ret.get(i, j) = mtx.get(i, index);
				}
			}
		}

		return ret;
	}
}

#endif // _NUMCPP_TPP_
