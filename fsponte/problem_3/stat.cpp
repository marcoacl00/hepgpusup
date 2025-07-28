#include "stat.hpp"

// Dependencies
#include <cstdlib>

type_t drand48(const type_t min, const type_t max) noexcept(false)
{
	if (min >= max)
		throw "Invalid range (min >= max)";

	return (max - min) * drand48() + min;
}

type_t coeff_det(const set_t dataset, const set_t fit) noexcept(false)
{
	const unsigned long DIM = dataset.size();

	if (DIM != fit.size())
		throw "Sets with different dimensions";

    type_t mean = 0;

	// Calculate mean
	{
		for (auto value : dataset)
			mean += value;

		mean /= DIM;
	}

    type_t
		ss_res = 0,
    	ss_tot = 0;

    for (unsigned long i = 0; i < DIM; ++i)
	{
        ss_res += (dataset[i] - fit[i]) * (dataset[i] - fit[i]);
        ss_tot += (dataset[i] - mean) * (dataset[i] - mean);
    }

    return (ss_tot == 0) ? 0 : 1 - (ss_res / ss_tot);
}

params_t coeff_det_grad(const set_t dataset, const set_t fit, const set_t fit_der_dept, const set_t fit_der_zero) noexcept(false)
{
	const unsigned long DIM = dataset.size();

	if (DIM != fit.size())
		throw "Sets with different dimensions";
	
	type_t mean = 0;

	// Calculate mean
	{
		for (auto value : dataset)
			mean += value;

		mean /= DIM;
	}

	type_t
		ss_res = 0,
    	ss_tot = 0,
		ss_der_dept = 0,
		ss_der_zero = 0;

	for (unsigned long i = 0; i < DIM; ++i)
	{
		ss_res += dataset[i] - fit[i];
		ss_tot += (dataset[i] - mean) * (dataset[i] - mean);
		ss_der_dept += fit_der_dept[i];
		ss_der_zero += fit_der_zero[i];
	}

	if (ss_tot == 0)
		return params_t{0, 0};
	
	type_t coeff = 2 * ss_res / ss_tot;

	return params_t{coeff * ss_der_dept, coeff * ss_der_zero};
}
