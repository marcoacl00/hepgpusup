#ifndef _HEADER_HPP_
#define _HEADER_HPP_

// Dependencies
#include <cmath>
#include <string>
#include <vector>

using type_t = long double; // Data type

typedef std::vector<std::string> file_lines_t; // Lines of a file
typedef std::vector<std::vector<type_t>> table_t; // Matrix of numerical values
typedef std::vector<type_t> set_t; // Set of values

// Parameters for the Lennard-Jones potential
struct params_t
{
	type_t
		dept, // Dept of the potencial (epsilon)
		zero; // Zero of the potencial (sigma)

	type_t norm(void)
	{
		return std::sqrt(this->dept * this->dept + this->zero * this->zero);
	}

	bool operator != (const params_t p)
	{
		return (this->dept != p.dept || this->zero != p.zero) ? true : false;
	}

	void operator -= (const params_t p)
	{
		this->dept -= p.dept;
		this->zero -= p.zero;
	}

	void operator *= (const type_t scalar)
	{
		this->dept *= scalar;
		this->zero *= scalar;
	}

	void operator /= (const type_t scalar)
	{
		this->dept /= scalar;
		this->zero /= scalar;
	}
};

constexpr const char csv_file_path[] = "../../../problems/lennard_jones.csv"; // File path

#endif // _HEADER_HPP_
