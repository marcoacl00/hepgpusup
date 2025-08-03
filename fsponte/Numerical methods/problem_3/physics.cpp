// Header file
#include "physics.hpp"

type_t lj_potential(const params_t params, const type_t radius)
{
	return 4 * params.dept * (std::pow(params.zero / radius, 12) - std::pow(params.zero / radius, 6));
}

type_t lj_der_dept(const params_t params, const type_t radius)
{
	return 4 * (std::pow(params.zero / radius, 12) - std::pow(params.zero / radius, 6));
}

type_t lj_der_zero(const params_t params, const type_t radius)
{
	return 4 * params.dept * ((12 * std::pow(params.zero, 11) / std::pow(radius, 12)) - (6 * std::pow(params.zero, 5) / std::pow(radius, 6)));
}

type_t harmonic_oscillator(const params_t params, const type_t radius, const type_t radius_min, const type_t spring_constant)
{
	return lj_potential(params, radius_min) + 0.5 * spring_constant * (radius - radius_min) * (radius - radius_min);
}
