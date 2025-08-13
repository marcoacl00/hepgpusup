#ifndef _STAT_HPP_
#define _STAT_HPP_

// Dependencies
#include "types.hpp"

/**
 * @brief Compute the mean of a vector
 * @param vec Vector
 * @return Mean of the components
*/
double mean(vector_t) noexcept(true);

/**
 * @brief Compute the mean of a matrix
 * @param fld Field
 * @return Mean of the elements
*/
double mean(field_t) noexcept(true);

#endif // _STAT_HPP_
