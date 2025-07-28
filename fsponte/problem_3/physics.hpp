#ifndef _PHYSICS_HPP_
#define _PHYSICS_HPP_

// Dependencies
#include "header.hpp"

/**
 * @brief Lennard-Jones potential
 * @param p Parameters
 * @param r Radius
 * @return Value of the potential
*/
type_t lj_potential(params_t, type_t);

/**
 * @brief Lennard-Jones potential partial derivative
 * @param p Parameters
 * @param r Radius
 * @return Derivative with respect to the dept
*/
type_t lj_der_dept(params_t, type_t);

/**
 * @brief Lennard-Jones potential partial derivative
 * @param p Parameters
 * @param r Radius
 * @return Derivative with respect to the zero
*/
type_t lj_der_zero(params_t, type_t);

/**
 * @brief Harmonic oscillator
 * @param x Position
 * @param m Mass
 * @param w Angular velocity
 * @return Potential
*/
type_t harmonic_oscillator(type_t, type_t, type_t);

#endif // _PHYSICS_HPP_
