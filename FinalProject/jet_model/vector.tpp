#ifndef _VECTOR_TPP_
#define _VECTOR_TPP_

template <typename type_t, unsigned long DIM>
vector_t<type_t, DIM>::vector_t(const vector_t<type_t, DIM>& vec)
{
	*this = vec;
}

template <typename type_t, unsigned long DIM>
constexpr unsigned long vector_t<type_t, DIM>::dim(void) const noexcept(true)
{
	return DIM;
}

template <typename type_t, unsigned long DIM>
void vector_t<type_t, DIM>::operator = (const vector_t<type_t, DIM>& vec) noexcept(true)
{
	for (unsigned long i = 0; i < DIM; ++i)
		this->_data[i] = vec[i];
}

template <typename type_t, unsigned long DIM>
inline type_t& vector_t<type_t, DIM>::operator [] (const unsigned long index) noexcept(false)
{
	if (index >= DIM)
		throw "Index is out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long DIM>
inline const type_t& vector_t<type_t, DIM>::operator [] (const unsigned long index) const noexcept(false)
{
	if (index >= DIM)
		throw "Index is out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long DIM>
vector_t<type_t, DIM> vector_t<type_t, DIM>::operator + (const vector_t<type_t, DIM>& vec) noexcept(true)
{
	vector_t<type_t, DIM> ret; // Return vector

	for (unsigned long i = 0; i < DIM; ++i)
		ret[i] = this->_data[i] + vec[i];

	return ret;
}

template <typename type_t, unsigned long DIM>
vector_t<type_t, DIM> vector_t<type_t, DIM>::operator - (const vector_t<type_t, DIM>& vec) noexcept(true)
{
	vector_t<type_t, DIM> ret; // Return vector

	for (unsigned long i = 0; i < DIM; ++i)
		ret[i] = this->_data[i] - vec[i];

	return ret;
}

template <typename type_t, unsigned long DIM>
vector_t<type_t, DIM> vector_t<type_t, DIM>::operator * (const type_t scalar) noexcept(true)
{
	vector_t<type_t, DIM> ret; // Return vector

	for (unsigned long i = 0; i < DIM; ++i)
		ret[i] = this->_data[i] * scalar;

	return ret;
}

template <typename type_t, unsigned long DIM>
vector_t<type_t, DIM> vector_t<type_t, DIM>::operator / (const type_t scalar) noexcept(true)
{
	vector_t<type_t, DIM> ret; // Return vector

	for (unsigned long i = 0; i < DIM; ++i)
		ret[i] = this->_data[i] / scalar;

	return ret;
}

template <typename type_t, unsigned long DIM>
type_t vector_t<type_t, DIM>::operator * (const vector_t<type_t, DIM>& vec) noexcept(true)
{
	type_t ret = 0; // Return scalar

	for (unsigned long i = 0; i < DIM; ++i)
		ret += this->_data[i] * vec[i];

	return ret;
}

#endif // _VECTOR_TPP_
