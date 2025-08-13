// Header file
#include "logging.hpp"

// Dependencies
#include <iostream>

void print(const vector_t vec) noexcept(true)
{
	const unsigned long DIM = vec.size();
	std::cout << '(' << vec[0];

	for (unsigned long i = 1; i < DIM; ++i)
		std::cout << ", " << vec[i];

	std::cout << ")\n";
}

void print(const field_t fld) noexcept(true)
{
	const unsigned long DIM = fld.size();
	std::cout << "{\n";

	for (unsigned long i = 0; i < DIM; ++i)
		print(fld[i]);

	std::cout << "}\n";
}
