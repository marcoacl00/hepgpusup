#ifndef _MATRIX_TPP_
#define _MATRIX_TPP_

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL>::matrix_t(const matrix_t<type_t, N_LIN, N_COL>& mtx)
{
	*this = mtx;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
constexpr unsigned long matrix_t<type_t, N_LIN, N_COL>::n_lin(void) const noexcept(true)
{
	return N_LIN;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
constexpr unsigned long matrix_t<type_t, N_LIN, N_COL>::n_col(void) const noexcept(true)
{
	return N_COL;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
constexpr unsigned long matrix_t<type_t, N_LIN, N_COL>::size(void) const noexcept(true)
{
	return N_LIN * N_COL;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void matrix_t<type_t, N_LIN, N_COL>::operator = (const matrix_t<type_t, N_LIN, N_COL>& mtx) noexcept(true)
{
	for (unsigned long i = 0; i < this->size(); ++i)
		this->_data[i] = mtx[i];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void matrix_t<type_t, N_LIN, N_COL>::operator = (const std::vector<type_t>& mtx) noexcept(false)
{
	if (this->size() != mtx.size())
		throw "Dimensions are different";

	for (unsigned long i = 0; i < this->size(); ++i)
		this->_data[i] = mtx[i];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
inline type_t& matrix_t<type_t, N_LIN, N_COL>::operator [] (const unsigned long index) noexcept(false)
{
	if (index >= N_LIN * N_COL)
		throw "Index is out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
inline const type_t& matrix_t<type_t, N_LIN, N_COL>::operator [] (const unsigned long index) const noexcept(false)
{
	if (index >= N_LIN * N_COL)
		throw "Index is out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
inline type_t& matrix_t<type_t, N_LIN, N_COL>::get(const unsigned long lin, const unsigned long col) noexcept(false)
{
	if (lin >= N_LIN || col >= N_COL)
		throw "Indexes are out of bounds";

	return this->_data[lin * N_COL + col];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
inline type_t matrix_t<type_t, N_LIN, N_COL>::get(const unsigned long lin, const unsigned long col) const noexcept(false)
{
	if (lin >= N_LIN || col >= N_COL)
		throw "Indexes are out of bounds";

	return this->_data[lin * N_COL + col];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator + (const matrix_t<type_t, N_LIN, N_COL>& mtx_1, const matrix_t<type_t, N_LIN, N_COL>& mtx_2) noexcept(true)
{
	matrix_t<type_t, N_LIN, N_COL> ret; // Return matrix

	for (unsigned long i = 0; i < mtx_1.size(); ++i)
		ret[i] = mtx_1[i] + mtx_2[i];

	return ret;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator - (const matrix_t<type_t, N_LIN, N_COL>& mtx_1, const matrix_t<type_t, N_LIN, N_COL>& mtx_2) noexcept(true)
{
	matrix_t<type_t, N_LIN, N_COL> ret; // Return matrix

	for (unsigned long i = 0; i < mtx_1.size(); ++i)
		ret[i] = mtx_1[i] - mtx_2[i];

	return ret;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator * (const matrix_t<type_t, N_LIN, N_COL>& mtx, const type_t scalar) noexcept(true)
{
	matrix_t<type_t, N_LIN, N_COL> ret; // Return matrix

	for (unsigned long i = 0; i < mtx.size(); ++i)
		ret[i] = mtx[i] * scalar;

	return ret;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator * (const type_t scalar, const matrix_t<type_t, N_LIN, N_COL>& mtx) noexcept(true)
{
	return mtx * scalar;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator / (const matrix_t<type_t, N_LIN, N_COL>& mtx, const type_t scalar) noexcept(true)
{
	matrix_t<type_t, N_LIN, N_COL> ret; // Return matrix

	for (unsigned long i = 0; i < mtx.size(); ++i)
		ret[i] = mtx[i] / scalar;

	return ret;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator / (const type_t scalar, const matrix_t<type_t, N_LIN, N_COL>& mtx) noexcept(true)
{
	return mtx / scalar;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator * (const matrix_t<type_t, N_LIN, N_COL>& mtx_1, const matrix_t<type_t, N_LIN, N_COL>& mtx_2) noexcept(true)
{
	matrix_t<type_t, N_LIN, N_COL> ret; // Return matrix

	for (unsigned long i = 0; i < N_LIN; ++i)
	{
		for (unsigned long j = 0; j < N_COL; ++j)
		{
			for (unsigned long k = 0; k < N_COL; ++k)
				ret.get(i, j) = mtx_1.get(i, k) * mtx_2.get(k, j);
		}
	}

	return ret;
}

#endif // _MATRIX_TPP_
