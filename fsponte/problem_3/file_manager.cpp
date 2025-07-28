// Header file
#include "file_manager.hpp"

// Dependencies
#include <fstream>
#include <sstream>

file_lines_t read_csv(const std::string file_name) noexcept(false)
{
	std::ifstream file(file_name);

	if (!file.is_open())
		throw "Input file not open";

	if (!file.good())
		throw "Input file not good";

	std::string line;
	file_lines_t file_lines;

	std::getline(file, line); // Skip header line

	for (unsigned long i = 0; std::getline(file, line); ++i)
		file_lines.push_back(line);

	file.close();

	return file_lines;
}

table_t flines_to_table(const file_lines_t file_lines) noexcept(true)
{
	table_t table;

	set_t row;
	std::string cell;
	std::stringstream string_stream;

	for (auto line : file_lines)
	{
		row.clear();
		cell.clear();
		string_stream = std::stringstream(line);

		while (std::getline(string_stream, cell, ','))
			row.push_back(std::stod(cell));

		table.push_back(row);
	}

	return table;
}

void write_dat(const std::string file_name, const set_t x, const set_t y, const set_t error) noexcept(false)
{
	std::ofstream file(file_name);

	if (!file.is_open())
		throw "Output file not open";

	if (!file.good())
		throw "Output file not good";

	const unsigned long DIM = x.size();

	if (DIM != y.size() || DIM != error.size())
		throw "Different set dimensions";

	for (unsigned long i = 0; i < DIM; ++i)
	{
		file
			<< x[i] << ' '
			<< y[i] << ' '
			<< error[i] << '\n';
	}

	file.close();
}
