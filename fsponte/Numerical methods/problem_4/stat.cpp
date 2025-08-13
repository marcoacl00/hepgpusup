// Header file
#include "stat.hpp"

double mean(const vector_t vec) noexcept(true)
{
	double sum = 0; // Sum of all components

	for (auto component : vec)
		sum += component;

	return sum / vec.size();
}

double mean(const field_t fld) noexcept(true)
{
	double sum = 0; // Sum of all elements

	for (auto line : fld)
	{
		for (auto element : line)
			sum += element;
	}

	return sum / (fld.size() * fld.size());
}
