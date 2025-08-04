#ifndef _VECTOR_HPP_
#define _VECTOR_HPP_

/**
 * @brief Vector object
 * @tparam type_t Data type
 * @tparam DIM Number of components
*/
template <typename type_t, unsigned long DIM>
class vector
{
private:

	static_assert(DIM >= 2, "Invalid dimension");

	type_t _data[DIM]; // Components

public:

	/**
	 * @brief Default constructor
	 * @note Sets all components to zero
	*/
	vector(void);

	/**
	 * @brief Copy constructor
	 * @param vec Vector
	*/
	vector(const vector<type_t, DIM>&);

	/**
	 * @brief Assignment operator
	 * @param vec Vector
	*/
	void operator = (vector<type_t, DIM>);

	/**
	 * @brief Get operator
	 * @param ind Index
	 * @return Respective component
	 * @throw Index out of bounds
	*/
	type_t& operator [] (unsigned long) noexcept(false);

	/**
	 * @brief Get operator
	 * @param ind Index
	 * @return Respective component
	 * @throw Index out of bounds
	*/
	const type_t& operator [] (unsigned long) const noexcept(false);

	/**
	 * @brief Addition operator
	 * @param vec Vector
	 * @return Added vector
	*/
	vector<type_t, DIM> operator + (vector<type_t, DIM>);

	/**
	 * @brief Dot product
	 * @param vec Vector
	 * @return Scalar
	*/
	type_t operator * (vector<type_t, DIM>);

	/**
	 * @brief Print vector to terminal
	*/
	void print(void) const;
};

// Template file
#include "vector.tpp"

#endif // _VECTOR_HPP_
