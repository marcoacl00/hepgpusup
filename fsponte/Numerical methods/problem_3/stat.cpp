#include "stat.hpp"

// Dependencies
#include <cstdlib>

type_t drand48(const type_t min, const type_t max) noexcept(false)
{
	if (min >= max)
		throw "Invalid range (min >= max)";

	return (max - min) * drand48() + min;
}

type_t cost_fn(const set_t dataset, const set_t fit) noexcept(false)
{
	const unsigned long DIM = dataset.size();

	if (DIM != fit.size())
		throw "Sets with different dimensions";

	type_t sum = 0;

	for (unsigned long i = 0; i < DIM; ++i)
		sum += (fit[i] - dataset[i]) * (fit[i] - dataset[i]);

	return sum;
}

params_t cost_fn_grad(const set_t dataset, const set_t fit, const set_t fit_der_dept, const set_t fit_der_zero) noexcept(false)
{
	const unsigned long DIM = dataset.size();

	if (DIM != fit.size() || DIM != fit_der_dept.size() || DIM != fit_der_zero.size())
		throw "Sets with different dimensions";
	
	type_t
		var,
		sum_dept = 0,
		sum_zero = 0;

	for (unsigned long i = 0; i < DIM; ++i)
	{
		var = fit[i] - dataset[i];
		sum_dept += var * fit_der_dept[i];
		sum_zero += var * fit_der_zero[i];
	}

	return params_t{2 * sum_dept, 2 * sum_zero};
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
