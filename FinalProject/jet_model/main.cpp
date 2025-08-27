#include <iostream>
#include <fstream>
#include <cmath>

#include "numcpp.hpp"
#include "jet_model.hpp"

void print(const vector_t&);
void print(const matrix_t&);

int main(void)
{
	constexpr unsigned long
		Nx = 200, // X dimension
		Ny = 200, // Y dimension
		Lx = 10, // Lattice X dimension
		Ly = 10; // Lattice Y dimension
	constexpr float
		x0 = 20, // Initial X coordinate
		y0 = 80, // Initial Y coordinate
		dx = static_cast<float>(Lx) / Nx, // X diferencial
		dy = static_cast<float>(Ly) / Ny; // Y diferencial
	vector_t
		x = numcpp::linspace(0, Lx, Nx), // Values for X dimension
		y = numcpp::linspace(0, Ly, Ny); // Values for Y dimension
	matrix_t
		X(Nx, vector_t(Ny)), // Mesh for X
		Y(Nx, vector_t(Ny)), // Mesh for Y
		medium(Nx, vector_t(Ny)), // Mesh for the medium 
		jet(Nx, vector_t(Ny)); // Mesh for the jet
	snapshots_t snapshots; // Snapshots of the jet over time
	std::ofstream file; // Output file

	numcpp::meshgrid(x, y, X, Y);

	// Set initial medium
    for (unsigned long i = 0; i < Nx; ++i)
	{
        for (unsigned long j = 0; j < Ny; ++j)
            medium[i][j] = std::sin(M_PI * X[i][j]) * std::cos(M_PI * Y[i][j]);
    }

	// Write medium to a file
	{
		file.open("output/medium.dat");

		for (unsigned long j = 0; j < Ny; ++j)
		{
			for (unsigned long i = 0; i < Nx; ++i)
				file << medium[i][j] << ' ';

			file << '\n';
		}

		file.close();
	}

	// Create the medium
	{
		constexpr float
			sigma_x = 0.5, // Standard deviation in X
			sigma_y = 0.5, // Standard deviation in Y
			sigma_x_sq = 2 * sigma_x * sigma_x,
			sigma_y_sq = 2 * sigma_y * sigma_y;

		float x_pos, y_pos;

		// Set the medium as a 2D Guassian distribution
		for (unsigned long i = 0; i < Nx; ++i)
		{
			for (unsigned long j = 0; j < Ny; ++j)
			{
				x_pos = (X[i][j] - x0 * dx) * (X[i][j] - x0 * dx);
				y_pos = (Y[i][j] - y0 * dy) * (Y[i][j] - y0 * dy);
				jet[i][j] = std::exp(-(x_pos / sigma_x_sq + y_pos / sigma_y_sq));
			}
		}
	}

	// Snapshots
	{
		const unsigned long N_STEPS = 5E2; // Number of steps
		float
			vx = 1, // Velocity in X
			vy = 0, // Velocity in Y
			g = 0.5, // Coupling constant
			CFL = 0.3, // Courant-Friedrichs-Lewy condition
			dt = CFL * std::min(
				(vx != 0.0) ? dx / std::abs(vx) : INF,
				(vy != 0.0) ? dy / std::abs(vy) : INF
			); // Time diferencial

		snapshots = evolve_jet(jet, medium, dt, dx, dy, vx, vy, g, N_STEPS);

		// Write snapshots to a file
		{
			file.open("output/snapshots.dat");

			for (unsigned long snap_index = 0; snap_index < N_STEPS; snap_index += N_STEPS / 10)
			{
				for (unsigned long j = 0; j < Ny; ++j)
				{
					for (unsigned long i = 0; i < Nx; ++i)
						file << snapshots[snap_index][i][j] << ' ';

					file << '\n';
				}

				file << '\n';
			}

			file.close();
		}
	}

	return 0;
}

void print(const vector_t& vec)
{
	for (const auto& element : vec)
		std::cout << element << ' ';

	std::cout << std::endl;
}

void print(const matrix_t& mtx)
{
	for (const auto& line : mtx)
	{
		for (const auto& element : line)
			std::cout << element << ' ';

		std::cout << '\n';
	}

	std::cout << std::flush;
}
