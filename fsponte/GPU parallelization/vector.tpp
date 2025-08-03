#ifndef _VECTOR_TPP_
#define _VECTOR_TPP_

template <typename type_t, unsigned long DIM>
vector<type_t, DIM>::vector(void)
{
	for (auto& component : this->_data)
		component = static_cast<type_t>(0);
}

template <typename type_t, unsigned long DIM>
vector<type_t, DIM>::vector(const vector<type_t, DIM>& vec)
{
	for (unsigned long i = 0; i < DIM; ++i)
		this->_data[i] = const_cast<type_t&>(vec[i]);
}

template <typename type_t, unsigned long DIM>
void vector<type_t, DIM>::operator = (vector<type_t, DIM> vec)
{
	for (unsigned long i = 0; i < DIM; ++i)
		this->_data[i] = vec[i];
}

template <typename type_t, unsigned long DIM>
type_t& vector<type_t, DIM>::operator [] (const unsigned long index) noexcept(false)
{
	if (index >= DIM)
		throw "Index out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long DIM>
const type_t& vector<type_t, DIM>::operator [] (const unsigned long index) const noexcept(false)
{
	if (index >= DIM)
		throw "Index out of bounds";

	return this->_data[index];
}

template <typename type_t, unsigned long DIM>
vector<type_t, DIM> vector<type_t, DIM>::operator + (vector<type_t, DIM> vec)
{
	vector<type_t, DIM> ret;

	for (unsigned long i = 0; i < DIM; ++i)
		ret[i] = this->_data[i] + vec[i];

	return ret;
}

template <typename type_t, unsigned long DIM>
type_t vector<type_t, DIM>::operator * (vector<type_t, DIM> vec)
{
	type_t ret = 0;

	for (unsigned long i = 0; i < DIM; ++i)
		ret += this->_data[i] * vec[i];

	return ret;
}

#endif // _VECTOR_TPP_
