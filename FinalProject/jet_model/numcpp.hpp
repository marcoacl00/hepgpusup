#ifndef _NUMCPP_HPP_
#define _NUMCPP_HPP_

// Dependencies
#include <cmath>
#include <limits>
#include <vector>

constexpr float inf = std::numeric_limits<float>::infinity(); // Infinity value

using vector_t = std::vector<float>;
using matrix_t = std::vector<vector_t>;
using snapshots_t = std::vector<matrix_t>;

// Axis in geometric space
enum axis_t
{
	X,
	Y
};

namespace numcpp
{
	/**
	 * @brief Create a evenly spaced array
	 * @param start Starting point
	 * @param stop Stoping point
	 * @param num Number of points
	 * @return Array
	 * @throw Invalid range
	 * @throw Invalid number of points
	*/
	vector_t linspace(float, float, unsigned long) noexcept(false);

	/**
	 * @brief Create a meshgrid
	 * @param x X vector
	 * @param y Y vector
	 * @param X X matrix
	 * @param Y Y matrix
	*/
	void meshgrid(const vector_t&, const vector_t&, matrix_t&, matrix_t&) noexcept(true);

	/**
	 * @brief Roll elements in matrix
	 * @param mtx Matrix
	 * @param a Amount to shift
	 * @param ax Axis to operate
	 * @return Shifted matrix
	*/
	matrix_t roll(const matrix_t&, unsigned long, axis_t) noexcept(true);
}

vector_t operator - (const vector_t&, const vector_t&) noexcept(true);

matrix_t operator - (const matrix_t&, const matrix_t&) noexcept(true);
matrix_t operator * (float, const matrix_t&) noexcept(true);
matrix_t operator / (const matrix_t&, float) noexcept(true);
matrix_t operator + (const matrix_t&, const matrix_t&) noexcept(true);
matrix_t operator * (const matrix_t&, const matrix_t&) noexcept(true);

#endif // _NUMCPP_HPP_
