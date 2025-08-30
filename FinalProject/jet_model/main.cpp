#include <iostream>
#include <fstream>
#include <cmath>

#include "vector.hpp"
#include "matrix.hpp"
#include "numcpp.hpp"
#include "jet_model.hpp"
#include "../PlasmaModel.hpp"

using type_t = float;

template <typename type_t, unsigned long DIM>
void print(const vector_t<type_t, DIM>&);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void print(const matrix_t<type_t, N_LIN, N_COL>&);

int main(void)
{
	constexpr unsigned long
		Nx = 200, // X dimension
		Ny = 200, // Y dimension
		Lx = 10, // Lattice X dimension
		Ly = 10; // Lattice Y dimension
	constexpr type_t
		x0 = 20, // Initial X coordinate
		y0 = 80, // Initial Y coordinate
		dx = static_cast<type_t>(Lx) / Nx, // X diferencial
		dy = static_cast<type_t>(Ly) / Ny; // Y diferencial
	vector_t<type_t, Nx> x = numcpp::linspace<type_t, Nx>(0, Lx); // Values for X dimension
	vector_t<type_t, Ny> y = numcpp::linspace<type_t, Ny>(0, Ly); // Values for Y dimension
	matrix_t<type_t, Nx, Ny>
		X, // Mesh for X
		Y, // Mesh for Y
		medium, // Mesh for the medium 
		jet; // Mesh for the jet
	std::ofstream file; // Output file

	numcpp::meshgrid(x, y, X, Y);

	{
		int
			D = 2,
			timeSteps = 1000,
			LeapSteps = 20;
		float
			a = 1,
			dt = 0.01,
			beta = 0.1,
			lambda = 0.2689,
			T = 0.01;
		PlasmaModel model(lambda, beta, Nx, a, D, timeSteps, dt, LeapSteps, T);
		model.InitializeGrid();
		model.RunSimulation();
		model.ExportData("output/medium.dat");

		// Set initial medium
		medium = model.GetEnergyField();
	}

	// Write medium to a file
	{
		file.open("output/medium.dat");

		for (unsigned long j = 0; j < medium.n_lin(); ++j)
		{
			for (unsigned long i = 0; i < medium.n_col(); ++i)
				file << medium.get(i, j) << ' ';

			file << '\n';
		}

		file.close();
	}

	// Create the medium
	{
		constexpr type_t
			sigma_x = 0.5, // Standard deviation in X
			sigma_y = 0.5, // Standard deviation in Y
			sigma_x_sq = 2 * sigma_x * sigma_x,
			sigma_y_sq = 2 * sigma_y * sigma_y;

		type_t x_pos, y_pos;

		// Set the medium as a 2D Guassian distribution
		for (unsigned long i = 0; i < Nx; ++i)
		{
			for (unsigned long j = 0; j < Ny; ++j)
			{
				x_pos = (X.get(i, j) - x0 * dx) * (X.get(i, j) - x0 * dx);
				y_pos = (Y.get(i, j) - y0 * dy) * (Y.get(i, j) - y0 * dy);
				jet.get(i, j) = std::exp(-(x_pos / sigma_x_sq + y_pos / sigma_y_sq));
			}
		}
	}

	// Snapshots
	{
		const unsigned long N_STEPS = 5E3; // Number of steps
		type_t
			vx = 1, // Velocity in X
			vy = 2, // Velocity in Y
			g = 0.5, // Coupling constant
			CFL = 0.3, // Courant-Friedrichs-Lewy condition
			dt = CFL * std::min(
				(vx != 0.0) ? dx / std::abs(vx) : INF,
				(vy != 0.0) ? dy / std::abs(vy) : INF
			); // Time diferencial
		std::vector<matrix_t<type_t, Nx, Ny>> snapshots = evolve_jet<type_t, Nx, Ny>(jet, medium, dt, dx, dy, vx, vy, g, N_STEPS); // Snapshots of the jet over time

		// Write snapshots to a file
		{
			file.open("output/snapshots.dat");

			for (unsigned long snap_index = 0; snap_index < N_STEPS; snap_index += N_STEPS / 100)
			{
				for (unsigned long j = 0; j < Ny; ++j)
				{
					for (unsigned long i = 0; i < Nx; ++i)
						file << snapshots[snap_index].get(i, j) << ' ';

					file << '\n';
				}

				file << '\n';
			}

			file.close();
		}
	}

	std::cout << std::endl;

	return 0;
}

template <typename type_t, unsigned long DIM>
void print(const vector_t<type_t, DIM>& vec)
{
	for (unsigned long i = 0; i < DIM; ++i)
		std::cout << vec[i] << ' ';

	std::cout << '\n';
}

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void print(const matrix_t<type_t, N_LIN, N_COL>& mtx)
{
	for (unsigned long i = 0; i < N_LIN; ++i)
	{
		for (unsigned long j = 0; j < N_COL; ++j)
			std::cout << mtx.get(i, j) << ' ';

		std::cout << '\n';
	}
}
