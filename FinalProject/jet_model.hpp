#ifndef _JET_MODEL_HPP_
#define _JET_MODEL_HPP_

// Dependencies
#include <vector>

// Spacial point of a scalar field
/**
 * @brief Spacial point of a scalar field
 * @tparam type_t Data type
 * @tparam N_DIM Number of dimensions
*/
template <typename type_t, unsigned long N_DIM>
struct point_t
{
	static_assert(N_DIM != 0, "Invalid number of dimensions");

	type_t position[N_DIM]; // Position in n-dimensional space
	type_t scalar; // Scalar value of the field
};

/**
 * @brief Field at a single time instant
 * @tparam type_t Data type
 * @tparam N_DIM Number of dimensions
*/
template <typename type_t, unsigned long N_DIM>
using field_t = std::vector<point_t<type_t, N_DIM>>;

/**
 * @brief Jet model over space-time
 * @tparam type_t Data type
 * @tparam N_DIM Number of dimensions
*/
template <typename type_t, unsigned long N_DIM>
class jet_t
{
private:

	std::vector<field_t<type_t, N_DIM>> _data; // All the time points of the spacial field

public:

	/**
	 * @brief Constructor
	*/
	jet_t(void);

	/**
	 * @brief Get operator
	 * @param t Time instant
	 * @return Field at time instant t 
	 * @throw Time instant is out of bounds
	*/
	field_t<type_t, N_DIM>& operator [] (unsigned long) noexcept(false);

	auto begin(void) const noexcept(true);
	auto end(void) const noexcept(true);
};

// Template file
#include "jet_model.tpp"

#endif // _JET_MODEL_HPP_
