#ifndef _NUMCPP_HPP_
#define _NUMCPP_HPP_

// Dependencies
#include <limits>
#include <vector>
#include "vector.hpp"
#include "matrix.hpp"

constexpr float INF = std::numeric_limits<float>::infinity(); // Infinity value

// Axis in geometric space
enum axis_t
{
	X, // X dimension
	Y // Y dimension
};

namespace numcpp
{
	/**
	 * @brief Create a evenly spaced array
	 * @param start Starting point
	 * @param stop Stoping point
	 * @return Array
	 * @throw Invalid range
	*/
	template <typename type_t, unsigned long DIM>
	vector_t<type_t, DIM> linspace(type_t, type_t) noexcept(false);

	/**
	 * @brief Create a meshgrid
	 * @param x X vector
	 * @param y Y vector
	 * @param X X matrix
	 * @param Y Y matrix
	*/
	template <typename type_t, unsigned long DIM>
	void meshgrid(const vector_t<type_t, DIM>&, const vector_t<type_t, DIM>&, matrix_t<type_t, DIM, DIM>&, matrix_t<type_t, DIM, DIM>&) noexcept(true);

	/**
	 * @brief Roll elements in matrix
	 * @param mtx Matrix
	 * @param a Amount to shift
	 * @param ax Axis to operate
	 * @return Shifted matrix
	*/
	template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
	matrix_t<type_t, N_LIN, N_COL> roll(const matrix_t<type_t, N_LIN, N_COL>&, unsigned long, axis_t) noexcept(true);
}

// Template file
#include "numcpp.tpp"

#endif // _NUMCPP_HPP_
