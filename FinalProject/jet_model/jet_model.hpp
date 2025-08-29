#ifndef _JET_MODEL_HPP_
#define _JET_MODEL_HPP_

// Dependencies
#include "numcpp.hpp"

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
using snapshots_t = std::vector<matrix_t<type_t, N_LIN, N_COL>>;

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
matrix_t<type_t, N_LIN, N_COL> upwind_deriv(const matrix_t<type_t, N_LIN, N_COL>&, type_t, type_t, axis_t);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
std::vector<matrix_t<type_t, N_LIN, N_COL>> evolve_jet(const matrix_t<type_t, N_LIN, N_COL>&, const matrix_t<type_t, N_LIN, N_COL>&, type_t, type_t, type_t, type_t, type_t, type_t, unsigned long);

// Template file
#include "jet_model.tpp"

#endif // _JET_MODEL_HPP_
