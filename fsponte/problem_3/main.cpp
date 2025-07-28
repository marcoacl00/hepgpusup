#include <iostream>

#include "header.hpp"
#include "stat.hpp"
#include "file_manager.hpp"
#include "physics.hpp"

int main()
{
	auto file_lines = read_csv(csv_file_path); // Read csv file
	auto table = flines_to_table(file_lines); // Convert file lines to table of values

	set_t radius, potential, fit, error;
	params_t params;

	// Create dataset
	{
		for (auto row : table)
		{
			radius.push_back(row[0]);
			error.push_back(row[1]);
			potential.push_back(row[2]);
		}

		write_dat("dataset.dat", radius, potential, error); // Write dataset file
	}

	// Initial parameter values
	{
		const unsigned long DIM = radius.size();
		params = params_t{potential[0], radius[0]};

		for (unsigned long i = 1; i < DIM; ++i)
		{
			if (params.dept <= potential[i])
				continue;

			params = params_t{potential[i], radius[i]};
		}

		params.dept *= -1;
	}

	// Calculate parameters
	{
		type_t
			cost, // Cost function
			coeff; // Coefficient of determination (R squared)
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
			cost = cost_fn(potential, fit);
			
			std::cout
				<< "R^2 = " << coeff << '\n'
				<< "Cost = " << cost << '\n';

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

			params_step = cost_fn_grad(potential, fit, fit_der_dept, fit_der_zero);
			params_step /= params_step.norm();

			if (params_step != params_step) // Check for nan
			{
				std::cout
					<< "nan detected\n"
					<< std::endl;
				
				params = params_t
				{
					drand48(1, 10),
					drand48(1, 5)
				};

				continue;
			}

			params_step *= 1E-2; // Learning rate
			
			std::cout
				<< "params_step = ("
				<< params_step.dept << ", "	<< params_step.zero
				<< ")\n";

			params -= params_step;
			
			std::cout << std::endl;
		}

		write_dat("fit.dat", radius, fit); // Write fit file
	}

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
