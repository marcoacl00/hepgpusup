#ifndef _STAT_HPP_
#define _STAT_HPP_

// Dependencies
#include "header.hpp"

/**
 * @brief Random number generator
 * @param min Minimum
 * @param max Maximum
 * @return Random number in the range
 * @throw Invalid range (min >= max)
*/
type_t drand48(type_t, type_t) noexcept(false);

/**
 * @brief Coefficient of determination
 * @param d Dataset
 * @param f Fit
 * @return R squared
 * @throw Sets with different dimensions
*/
type_t coeff_det(set_t, set_t) noexcept(false);

/**
 * @brief Gradient of the coefficient of determination
 * @param d Dataset
 * @param f Fit
 * @param fdd Fit derivative with respect to dept
 * @param fdz Fit derivative with respect to zero
 * @return Gradient of R squared
 * @throw Sets with different dimensions
*/
params_t coeff_det_grad(set_t, set_t, set_t, set_t) noexcept(false);

#endif // _STAT_HPP_
