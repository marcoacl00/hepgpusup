// Header file
#include "types.hpp"

// Dependencies
#include <cstdlib>

vector_t init_field(const unsigned long DIM) noexcept(false)
{
	if (DIM <= 1)
		throw "Invalid dimenion";
	
	auto ret = vector_t(DIM);

	for (auto& component : ret)
		component = drand48();

	return ret;
}

field_t init_lattice(const unsigned long DIM) noexcept(false)
{
	if (DIM <= 1)
		throw "Invalid dimenion";

	return field_t(DIM, vector_t(DIM, 0.0));
}
