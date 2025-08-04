#ifndef _MATRIX_TPP_
#define _MATRIX_TPP_

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix<type_t, N_LIN, N_COL>::matrix(void)
{
	for (auto& line : this->_data)
		line = vector<type_t, N_COL>();
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix<type_t, N_LIN, N_COL>::matrix(const matrix<type_t, N_LIN, N_COL>& mtx)
{
	for (unsigned long i = 0; i < N_LIN; ++i)
		this->_data[i] = mtx[i];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void matrix<type_t, N_LIN, N_COL>::operator = (matrix<type_t, N_LIN, N_COL> mtx)
{
	for (unsigned long i = 0; i < N_LIN; ++i)
		this->_data[i] = mtx[i];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
vector<type_t, N_COL>& matrix<type_t, N_LIN, N_COL>::operator [] (const unsigned long index) noexcept(false)
{
	if (index >= N_LIN)
		throw "Index out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
const vector<type_t, N_COL>& matrix<type_t, N_LIN, N_COL>::operator [] (const unsigned long index) const noexcept(false)
{
	if (index >= N_LIN)
		throw "Index out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix<type_t, N_LIN, N_COL> matrix<type_t, N_LIN, N_COL>::operator + (matrix<type_t, N_LIN, N_COL> mtx)
{
	matrix<type_t, N_LIN, N_COL> ret;

	for (unsigned long i = 0; i < N_LIN; ++i)
		ret[i] = this->_data[i] + mtx[i];

	return ret;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
template <unsigned long _N_COL>
matrix<type_t, N_LIN, _N_COL> matrix<type_t, N_LIN, N_COL>::operator * (matrix<type_t, N_COL, _N_COL> mtx)
{
	matrix<type_t, N_LIN, _N_COL> ret;

	for (unsigned long i = 0; i < N_LIN; ++i)
	{
		for (unsigned long j = 0; j < _N_COL; ++j)
		{
			ret[i][j] = 0;

			for (unsigned long k = 0; k < N_COL; ++k)
				ret[i][j] += this->_data[i][k] * mtx[k][j];
		}
	}

	return ret;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
vector<type_t, N_LIN> matrix<type_t, N_LIN, N_COL>::operator * (vector<type_t, N_COL> vec)
{
	vector<type_t, N_LIN> ret;

	for (unsigned long i = 0; i < N_LIN; ++i)
	{
		ret[i] = 0;

		for (unsigned long j = 0; j < N_COL; ++j)
			ret[i] += this->_data[i][j] * vec[j];
	}

	return ret;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix<type_t, N_COL, N_LIN> matrix<type_t, N_LIN, N_COL>::transpose(void) const
{
	matrix<type_t, N_COL, N_LIN> ret;

	for (unsigned long i = 0; i < N_LIN; ++i)
	{
		for (unsigned long j = 0; j < N_COL; ++j)
			ret[i][j] = this->_data[j][i];
	}

	return ret;
}

#endif // _MATRIX_TPP_
