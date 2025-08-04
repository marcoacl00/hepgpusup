#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

// Dependencies
#include "vector.hpp"

/**
 * @brief Matrix object
 * @tparam type_t Data type
 * @tparam N_LIN Number of lines
 * @tparam N_COL Number of columns
*/
template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
class matrix
{
private:

	static_assert(N_LIN >= 1, "Invalid number of lines");
	static_assert(N_COL >= 1, "Invalid number of columns");

	vector<type_t, N_COL> _data[N_LIN]; // Lines

public:

	/**
	 * @brief Default constructor
	 * @note Sets all elements to zero
	*/
	matrix(void);

	/**
	 * @brief Copy contructor
	 * @param mtx Matrix
	*/
	matrix(const matrix<type_t, N_LIN, N_COL>&);

	/**
	 * @brief Assigment operator
	 * @param mtx Matrix
	*/
	void operator = (matrix<type_t, N_LIN, N_COL>);

	/**
	 * @brief Get operator
	 * @param ind Index
	 * @return Respective line
	 * @throw Index out of bounds
	*/
	vector<type_t, N_COL>& operator [] (unsigned long) noexcept(false);

	/**
	 * @brief Get operator
	 * @param ind Index
	 * @return Respective line
	 * @throw Index out of bounds
	*/
	const vector<type_t, N_COL>& operator [] (unsigned long) const noexcept(false);

	/**
	 * @brief Addition operator
	 * @param mtx Matrix
	 * @return Added matrix
	*/
	matrix<type_t, N_LIN, N_COL> operator + (matrix<type_t, N_LIN, N_COL>);

	/**
	 * @brief Multiplication operator
	 * @tparam _N_COL New number of columns
	 * @param mtx Matrix
	 * @return Multiplied matrix
	*/
	template <unsigned long _N_COL>
	matrix<type_t, N_LIN, _N_COL> operator * (matrix<type_t, N_COL, _N_COL>);

	/**
	 * @brief Multiplication operator
	 * @param vec Vector
	 * @return Transformed vector
	*/
	vector<type_t, N_LIN> operator * (vector<type_t, N_COL>);

	/**
	 * @brief Transpose
	 * @return Transposed matrix
	*/
	matrix<type_t, N_COL, N_LIN> transpose(void) const;
};

// Template file
#include "matrix.tpp"

#endif // _MATRIX_HPP_
