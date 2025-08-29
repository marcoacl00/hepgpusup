#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
class matrix_t
{
private:

	type_t _data[N_LIN * N_COL];

public:

	/**
	 * @brief Default constructor
	*/
	matrix_t(void) = default;

	/**
	 * @brief Copy constructor
	 * @param mtx Matrix
	*/
	matrix_t(const matrix_t<type_t, N_LIN, N_COL>&);

	constexpr unsigned long n_lin(void) const noexcept(true);
	constexpr unsigned long n_col(void) const noexcept(true);
	constexpr unsigned long size(void) const noexcept(true);

	void operator = (const matrix_t<type_t, N_LIN, N_COL>&) noexcept(true);
	inline type_t& operator [] (unsigned long) noexcept(false);
	inline const type_t& operator [] (unsigned long) const noexcept(false);
	inline type_t& get(unsigned long, unsigned long) noexcept(false);
	inline type_t get(unsigned long, unsigned long) const noexcept(false);
};

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator + (const matrix_t<type_t, N_LIN, N_COL>&, const matrix_t<type_t, N_LIN, N_COL>&) noexcept(true);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator - (const matrix_t<type_t, N_LIN, N_COL>&, const matrix_t<type_t, N_LIN, N_COL>&) noexcept(true);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator * (const matrix_t<type_t, N_LIN, N_COL>&, type_t) noexcept(true);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator * (type_t, const matrix_t<type_t, N_LIN, N_COL>&) noexcept(true);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator / (const matrix_t<type_t, N_LIN, N_COL>&, type_t) noexcept(true);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator / (type_t, const matrix_t<type_t, N_LIN, N_COL>&) noexcept(true);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> operator * (const matrix_t<type_t, N_LIN, N_COL>&, const matrix_t<type_t, N_LIN, N_COL>&) noexcept(true);

// Template file
#include "matrix.tpp"

#endif // _MATRIX_HPP_
