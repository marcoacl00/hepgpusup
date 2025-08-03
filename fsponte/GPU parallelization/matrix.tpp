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
void matrix<type_t, N_LIN, N_COL>::operator = (const matrix<type_t, N_LIN, N_COL>& mtx)
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
matrix<type_t, N_LIN, N_COL> matrix<type_t, N_LIN, N_COL>::operator + (const matrix<type_t, N_LIN, N_COL>& mtx)
{
	matrix<type_t, N_LIN, N_COL> ret;

	for (unsigned long i = 0; i < N_LIN; ++i)
		ret[i] = this->_data[i] + mtx[i];

	return ret;
}

#endif // _MATRIX_TPP_
