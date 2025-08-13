#ifndef _TYPES_HPP_
#define _TYPES_HPP_

// Dependencies
#include <vector>

using vector_t = std::vector<double>; // Vector
using field_t = std::vector<vector_t>; // Matrix

/**
 * @brief Initialize the field
 * @param DIM Number of components
 * @return Vector
 * @throw Invalid dimension
*/
vector_t init_field(const unsigned long) noexcept(false);

/**
 * @brief Initialize the lattice
 * @param DIM Number of lines and columns
 * @return Field
 * @throw Invalid dimension
 * @note Lattice is a square matrix
*/
field_t init_lattice(const unsigned long) noexcept(false);

#endif // _TYPES_HPP_
