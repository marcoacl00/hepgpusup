#ifndef _VECTOR_HPP_
#define _VECTOR_HPP_

template <typename type_t, unsigned long DIM>
class vector_t
{
private:

	type_t _data[DIM];

public:

	/**
	 * @brief Default constructor
	*/
	vector_t(void) = default;

	/**
	 * @brief Copy constructor
	 * @param vec Vector
	*/
	vector_t(const vector_t<type_t, DIM>&);

	/**
	 * @brief Get dimension
	 * @return Number of components
	*/
	constexpr unsigned long dim(void) const noexcept(true);

	/**
	 * @brief Assignment operator
	 * @param vec Vector
	 * @note Copies all components
	*/
	void operator = (const vector_t<type_t, DIM>&) noexcept(true);

	/**
	 * @brief Subscript operator
	 * @param ind Index
	 * @return Reference to indexed component
	 * @throw Index is out of bounds
	*/
	inline type_t& operator [] (unsigned long) noexcept(false);

	/**
	 * @brief Subscript operator
	 * @param ind Index
	 * @return Constant reference to indexed component
	 * @throw Index is out of bounds
	*/
	inline const type_t& operator [] (unsigned long) const noexcept(false);

	vector_t<type_t, DIM> operator + (const vector_t<type_t, DIM>&) noexcept(true);
	vector_t<type_t, DIM> operator - (const vector_t<type_t, DIM>&) noexcept(true);
	vector_t<type_t, DIM> operator * (type_t) noexcept(true);
	vector_t<type_t, DIM> operator / (type_t) noexcept(true);
	type_t operator * (const vector_t<type_t, DIM>&) noexcept(true);
};

// Template file
#include "vector.tpp"

#endif // _VECTOR_HPP_
