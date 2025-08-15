// Header file
#include "physics.hpp"

// Depedencies
#include <cmath>
#include "constexpr.hpp"

inline double kinetic_term(const double conj_mom) noexcept(true)
{
	return 0.5 * std::pow(conj_mom, 2);
}

double neighbor_interaction(const double mag, const vector_t n_mag, const double lat_spa) noexcept(true)
{
    double sum = 0; // Sum of all the neighbors

	for (const auto& n_mag : n_mags)
        sum += std::pow(mag - n_mag, 2);

	return 0.5 * sum / (a * a);
}

inline double quartic_magnetization(const double temp, const double mag) noexcept(true)
{
	return temp * std::pow(mag, 2);
}

inline double quatic_coupling(const double temp, const double mag) noexcept(true)
{
	return (1 - CRIT_TEMP / temp) * std::pow(mag, 4);
}

inline double external_field(const double mag, const double fld_strength) noexcept(true)
{
	return fld_strength * mag;
}

inline double site_energy
(
	const double kinetic_term,
    const double n_mag_term,
    const double quartic_mag_term,
	const double quartic_coup_term,
	const double ext_fld_term
) noexcept(true)
{
    return kinetic_term + n_mag_term + quartic_mag_term + quartic_coup_term + ext_fld_term;
}

double hamiltonian(const double lat_spa, const unsigned long num_dims, const vector_t energy) noexcept(true)
{
	double sum = 0; // Sum of all energy terms

	for (const auto& term : energy)
		sum += term;

	return std::pow(lat_spa, num_dims) * sum;
}
