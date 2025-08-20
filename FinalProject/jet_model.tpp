#ifndef _JET_MODEL_TPP_
#define _JET_MODEL_TPP_

template <typename type_t, unsigned long N_DIM>
jet_t<type_t, N_DIM>::jet_t(void)
{
	this->_data.push_back(field_t<type_t, N_DIM>()); // First instant
	this->_data[0].push_back(point_t<type_t, N_DIM>()); // Field in the first instant
}

template <typename type_t, unsigned long N_DIM>
field_t<type_t, N_DIM>& jet_t<type_t, N_DIM>::operator [] (const unsigned long t) noexcept(false)
{
	if (t >= this->_data.size())
		throw "Time instant is out of bounds";

	return this->_data[t];
}

template <typename type_t, unsigned long N_DIM>
auto jet_t<type_t, N_DIM>::begin(void) const noexcept(true)
{ return this->_data.begin(); }

template <typename type_t, unsigned long N_DIM>
auto jet_t<type_t, N_DIM>::end(void) const noexcept(true)
{ return this->_data.end(); }

#endif // _JET_MODEL_TPP_
