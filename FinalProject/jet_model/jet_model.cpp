// Header file
#include "jet_model.hpp"

matrix_t upwind_deriv(const matrix_t& field, const float diferencial, const float velocity, const axis_t axis)
{
	if (velocity >= 0)
		return (field - numcpp::roll(field, 1, axis)) / diferencial;

	return (numcpp::roll(field, -1, axis) - field) / diferencial;
}

snapshots_t evolve_jet
(
	const matrix_t& jet_0, const matrix_t& medium,
	const float dt, const float dx, const float dy,
	const float vx, const float vy,
	const float g,
	const unsigned long N_STEPS
)
{
	matrix_t jet = jet_0;

	snapshots_t snapshots(N_STEPS, matrix_t(jet_0.size(), vector_t(jet_0[0].size(), 0)));

	for (unsigned long i = 0; i < N_STEPS; ++i)
	{
		jet = jet - dt * (g * medium * jet + vx * upwind_deriv(jet, dx, vx, axis_t::X) + vy * upwind_deriv(jet, dy, vy, axis_t::Y));
		snapshots[i] = jet;
	}

	return snapshots;
}
