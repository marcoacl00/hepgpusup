#include <iostream>
#include <iomanip>
#include <fstream>

#include "vector.hpp"
#include "matrix.hpp"
#include "numcpp.hpp"
#include "jet_model.hpp"
#include "../PlasmaModel.hpp"

template <typename type_t, unsigned long DIM>
void print(const vector_t<type_t, DIM>&);

template <typename type_t, unsigned long N_LIN, unsigned long N_COL>
void print(const matrix_t<type_t, N_LIN, N_COL>&);

int main(void)
{
	constexpr unsigned long N = 200; // X and Y dimensions
	constexpr float
		x0 = 20, // Initial X coordinate
		y0 = 80, // Initial Y coordinate
		dx = 0.1, // X diferencial
		dy = 0.1; // Y diferencial
	float
		g = 1, // Coupling constant
		dt = 0.005; // Time diferencial
	vector_t<float, N>
		x = numcpp::linspace<float, N>(0, N / 10), // Values for X dimension
		y = numcpp::linspace<float, N>(0, N / 10); // Values for Y dimension
	matrix_t<float, N, N>
		X, // Mesh for X
		Y, // Mesh for Y
		medium, // Mesh for the medium 
		jet; // Mesh for the jet
	std::ofstream file; // Output file

	std::cout << std::fixed << std::setprecision(10);
	numcpp::meshgrid(x, y, X, Y);

	// Create the medium
	{
		int
			D = 2,
			timeSteps = 5000,
			LeapSteps = 20;
		float
			a = 1,
			mth = 1,
			T = 0.8;

		PlasmaModel model(N, a, D, timeSteps, dt, LeapSteps, mth, g, T);
		model.InitializeGrid();
		model.RunSimulation();
		model.ExportData("output/medium.dat");

		// Set initial medium
		medium = model.GetEnergyField();
	}

	// Create the initial conditions of the jet
	{
		constexpr float
			sigma_x = 0.5, // Standard deviation in X
			sigma_y = 0.5, // Standard deviation in Y
			sigma_x_sq = 2 * sigma_x * sigma_x,
			sigma_y_sq = 2 * sigma_y * sigma_y;

		float x_pos, y_pos;

		// Set the jet as a 2D Guassian distribution
		for (unsigned long i = 0; i < N; ++i)
		{
			for (unsigned long j = 0; j < N; ++j)
			{
				x_pos = (X.get(i, j) - x0 * dx) * (X.get(i, j) - x0 * dx);
				y_pos = (Y.get(i, j) - y0 * dy) * (Y.get(i, j) - y0 * dy);
				jet.get(i, j) = std::exp(-(x_pos / sigma_x_sq + y_pos / sigma_y_sq));
			}
		}
	}

	// Compute the snapshots of the jet
	{
		constexpr float TIME_RANGE = 10; // Time duration of the jet simulation
		const auto N_STEPS = static_cast<unsigned long>(TIME_RANGE / dt); // Number of steps
		float
			vx = 1, // Velocity in X
			vy = 2; // Velocity in Y
		std::vector<matrix_t<float, N, N>> snapshots = evolve_jet<float, N, N>(jet, medium, dt, dx, dy, vx, vy, g, N_STEPS); // Snapshots of the jet over time

		// Write snapshots to a file
		{
			file.open("output/snapshots.dat");
			file.precision(10);

			for (unsigned long snap_index = 0; snap_index < N_STEPS; snap_index += N_STEPS / 100)
			{
				for (unsigned long j = 0; j < N; ++j)
				{
					for (unsigned long i = 0; i < N; ++i)
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
