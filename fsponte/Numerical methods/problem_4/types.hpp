#ifndef _TYPES_HPP_
#define _TYPES_HPP_

// Dependencies
#include <vector>

using vector_t = std::vector<double>; // Vector
using field_t = std::vector<vector_t>; // Matrix

/**
 * @brief Initialize the field
 * @param DIM Number of components
 * @param val Initial value
 * @return Field
 * @throw Invalid dimension
*/
field_t init_field(unsigned long, double) noexcept(false);

/**
 * @brief Initialize the lattice
 * @param DIM Number of lines and columns
 * @return Lattice
 * @throw Invalid dimension
 * @note Lattice is a square matrix with all elements set to zero
*/
field_t init_lattice(unsigned long) noexcept(false);

#endif // _TYPES_HPP_
