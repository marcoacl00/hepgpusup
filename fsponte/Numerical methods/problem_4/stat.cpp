// Header file
#include "stat.hpp"

double mean(const vector_t vec) noexcept(true)
{
	double sum = 0; // Sum of all components

	for (const auto component : vec)
		sum += component;

	return sum / vec.size();
}

double mean(const field_t fld) noexcept(true)
{
	double sum = 0; // Sum of all elements

	for (const auto line : fld)
	{
		for (const auto element : line)
			sum += element;
	}

	return sum / (fld.size() * fld.size());
}
