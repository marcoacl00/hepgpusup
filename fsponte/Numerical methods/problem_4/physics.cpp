// Header file
#include "physics.hpp"

// Depedencies
#include <cmath>
#include "constexpr.hpp"

double kinetic_energy(const double conj_mom) noexcept(true)
{
	return 0.5 * conj_mom * conj_mom;
}

double neighbor_interaction(const double mag, const vector_t n_mag, const double lat_spa) noexcept(true)
{
    double sum = 0; // Sum of all the neighbors

	for (const auto& neighbor : n_mag)
        sum += (mag - neighbor) * (mag - neighbor);

	return 0.5 * sum / (lat_spa * lat_spa);
}

double quartic_magnetization(const double temp, const double mag) noexcept(true)
{
	return temp * mag * mag;
}

double quartic_coupling(const double temp, const double mag) noexcept(true)
{
	return (1 - CRIT_TEMP / temp) * mag * mag * mag * mag;
}

double external_field(const double mag, const double fld_strength) noexcept(true)
{
	return fld_strength * mag;
}

double hamiltonian(const double lat_spa, const unsigned long num_dims, const vector_t energy) noexcept(true)
{
	double sum = 0; // Sum of all energy terms

	for (const auto& term : energy)
		sum += term;

	return std::pow(lat_spa, num_dims) * sum;
}
