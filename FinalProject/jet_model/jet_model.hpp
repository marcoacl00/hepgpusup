#ifndef _JET_MODEL_HPP_
#define _JET_MODEL_HPP_

// Dependencies
#include "numcpp.hpp"

matrix_t upwind_deriv(const matrix_t&, float, float, axis_t);
snapshots_t evolve_jet(const matrix_t&, const matrix_t&, float, float, float, float, float, float, unsigned long);

#endif // _JET_MODEL_HPP_
