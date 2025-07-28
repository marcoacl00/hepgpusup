#include <iostream>

#include "header.hpp"
#include "stat.hpp"
#include "file_manager.hpp"
#include "lj_potential.hpp"

int main()
{
	auto file_lines = read_csv(csv_file_path); // Read csv file
	auto table = flines_to_table(file_lines); // Convert file lines to table of values

	set_t radius, potential, fit, error;
	params_t params = {0, 0};

	// Create dataset
	{
		for (auto row : table)
		{
			radius.push_back(row[0]);
			error.push_back(row[1]);
			potential.push_back(row[2]);
		}
	}

	write_dat("dataset.dat", radius, potential, error); // Write dataset file

	// Calculate parameters
	{
		type_t coeff; // Coefficient of determination (R squared)
		set_t fit_der_dept, fit_der_zero;
		params_t params_step;

		for (unsigned long i = 1; i <= 1E3; ++i)
		{
			std::cout
				<< "Iteration: " << i << '\n'
				<< "params = (" << params.dept << ", " << params.zero << ")\n";

			// Calculate the fit
			{
				fit.clear();

				for (auto r : radius) // Create fit
					fit.push_back(lj_potential(params, r));
			}

			coeff = coeff_det(potential, fit);
			std::cout << "R^2 = " << coeff << '\n';
			
			if (coeff <= 0 || coeff > 1) // Invalid coefficient
			{
				params = params_t
				{
					drand48(1, 10),
					drand48(1, 5)
				};

				std::cout << std::endl;

				continue;
			}

			if (coeff >= 0.98) // Exit condition
			{
				std::cout << std::endl;

				break;
			}

			// Calculate the fit derivatives
			{
				fit_der_dept.clear();
				fit_der_zero.clear();

				for (auto r : radius)
				{
					fit_der_dept.push_back(lj_der_dept(params, r));
					fit_der_zero.push_back(lj_der_zero(params, r));
				}
			}

			params_step = coeff_det_grad(potential, fit, fit_der_dept, fit_der_zero);
			params_step /= params_step.norm();
			
			std::cout
				<< "params_step = ("
				<< params_step.dept << ", " << params_step.zero << ")\n"
				<< std::endl;

			params -= params_step;
		}
	}

	write_dat("fit.dat", radius, fit, error); // Write fit file

	// Print information
	{
		std::cout
			<< "LJ Parameters:" << '\n'
			<< '\t' << "Dept = " << params.dept << '\n'
			<< '\t' << "Zero = " << params.zero << '\n'
			<< '\n'
			<< "R^2 = " << coeff_det(potential, fit) << '\n'
			<< std::endl;
	}

	return 0;
}
