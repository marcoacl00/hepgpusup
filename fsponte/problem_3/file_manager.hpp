#ifndef _FILE_MANAGER_HPP_
#define _FILE_MANAGER_HPP_

// Dependencies
#include "header.hpp"

/**
 * @brief Read a csv file
 * @param f_nam File name
 * @return Lines of the file
 * @throw Input file not open
 * @throw Input file not good
*/
file_lines_t read_csv(const std::string) noexcept(false);

/**
 * @brief Convert file lines to table
 * @param f_lines File lines
 * @return Table of values
*/
table_t flines_to_table(const file_lines_t) noexcept(true);

/**
 * @brief Write dataset
 * @param f_nam File name
 * @param x X values
 * @param y Y values
 * @param er Error values
 * @throw Output file not open
 * @throw Output file not good
 * @throw Different set dimensions
*/
void write_dat(const std::string, const set_t, const set_t, const set_t) noexcept(false);

#endif // _FILE_MANAGER_HPP_
