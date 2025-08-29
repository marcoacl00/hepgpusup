#ifndef _JET_MODEL_TPP_
#define _JET_MODEL_TPP_

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> upwind_deriv(const matrix_t<type_t, N_LIN, N_COL>& field, const type_t diferencial, const type_t velocity, const axis_t axis)
{
	if (velocity >= 0)
		return (field - numcpp::roll(field, 1, axis)) / diferencial;

	return (numcpp::roll(field, -1, axis) - field) / diferencial;
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
std::vector<matrix_t<type_t, N_LIN, N_COL>> evolve_jet
(
	const matrix_t<type_t, N_LIN, N_COL>& jet_0, const matrix_t<type_t, N_LIN, N_COL>& medium,
	const type_t dt, const type_t dx, const type_t dy,
	const type_t vx, const type_t vy,
	const type_t g, const unsigned long N_STEPS
)
{
	matrix_t<type_t, N_LIN, N_COL> jet = jet_0;
	std::vector<matrix_t<type_t, N_LIN, N_COL>> ret(N_STEPS); // Return snapshots

	for (unsigned long i = 0; i < jet.size(); ++i)
	{
		if (jet[i] != jet_0[i])
			std::cout << i << std::endl;
	}

	for (unsigned long i = 0; i < N_STEPS; ++i)
	{
		jet = jet - dt * (g * medium * jet + vx * upwind_deriv(jet, dx, vx, axis_t::X) + vy * upwind_deriv(jet, dy, vy, axis_t::Y));
		ret[i] = jet;
	}

	return ret;
}

#endif // _JET_MODEL_TPP_
