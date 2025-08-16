#ifndef _PHYSICS_HPP_
#define _PHYSICS_HPP_

// Dependencies
#include "types.hpp"

/**
 * @brief Kinetic term
 * @param conj_mom Conjugate momentum
 * @return Kinetic term value
*/
inline double kinetic_energy(double) noexcept(true);

/**
 * @brief Immediate neighbor interaction term
 * @param mag Magnetization
 * @param n_mag Neighbor magnetizations
 * @param lat_spa Lattice spacing
 * @return Immediate neighbor interaction value
*/
double neighbor_interaction(double, vector_t, double) noexcept(true);

/**
 * @brief Quartic magnetization term
 * @param temp Temperature
 * @param mag Magnetization
 * @return Quatic magnetization value
*/
inline double quartic_magnetization(double, double) noexcept(true);

/**
 * @brief Quartic coupling term
 * @param temp Temperature
 * @return Quatic coupling value
*/
inline double quatic_coupling(double) noexcept(true);

/**
 * @brief External field term
 * @param mag Magnetization
 * @param h Field strength
 * @return External field value
*/
inline double external_field(double, double) noexcept(true);

/**
 * @brief Site energy term
 * @param k Kinetic term
 * @param n_mag Neighbor magnetization term
 * @param q_mag Quartic_mag_term
 * @param q_coup Quartic coupling term
 * @param ext External field term
 * @return Site energy value
*/
inline double site_energy(double, double, double, double, double) noexcept(true);

/**
 * @brief Pseudo-Hamiltonian of the system in a discretized lattice
 * @param lat_spa Lattice spacing
 * @param num_dims Number of dimensions
 * @param energy Energy terms
 * @return Pseudo-Hamiltonian value
*/
double hamiltonian(double, unsigned long, vector_t) noexcept(true);

#endif // _PHYSICS_HPP_
