// Header file
#include "jet_model.hpp"

matrix_t upwind_deriv(const matrix_t& E, const float v, const float d, const axis_t axis)
{
	if (v >= 0)
		return (E - numcpp::roll(E, 1, axis)) / d;

	return (numcpp::roll(E, -1, axis) - E) / d;
}

snapshots_t evolve_jet
(
	const matrix_t& E0, const matrix_t& medium,
	const float dt, const float dx, const float dy,
	const float vx, const float vy,
	const float g,
	const unsigned long N_STEPS
)
{
	matrix_t
		E = E0,
		dE_dx, dE_dy, loss;

	snapshots_t snapshots(N_STEPS, matrix_t(E0.size(), vector_t(E0[0].size(), 0)));

	for (unsigned long i = 0; i < N_STEPS; ++i)
	{
		dE_dx = upwind_deriv(E, vx, dx, axis_t::X);
		dE_dy = upwind_deriv(E, vy, dy, axis_t::Y);
		loss = - g * medium * E;

		E = E + dt * (loss - (vx * dE_dx + vy * dE_dy));
		snapshots[i] = E;
	}

	return snapshots;
}
