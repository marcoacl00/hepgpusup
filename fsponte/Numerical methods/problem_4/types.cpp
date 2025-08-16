// Header file
#include "types.hpp"

// Dependencies
#include <cstdlib>

field_t init_field(const unsigned long DIM, const double val) noexcept(false)
{
	if (DIM <= 1)
		throw "Invalid dimenion";
	
	auto ret = field_t(DIM);

	for (auto& line : ret)
	{
		for (auto& element : line)
			element = val;
	}

	return ret;
}

field_t init_lattice(const unsigned long DIM) noexcept(false)
{
	if (DIM <= 1)
		throw "Invalid dimenion";

	return field_t(DIM, vector_t(DIM, 0.0));
}
