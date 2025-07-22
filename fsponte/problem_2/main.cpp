#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

constexpr const unsigned long NUM_STEPS = 1E3;
constexpr const double dt = 1E-3;

struct params_t
{
	double
		alpha, // Growth rate of the preys
		beta, // Predation rate
		gamma, // Predator death rate
		theta; // Growth rate of predators per prey eaten
};

struct point_t
{
	double x, y;
};

double drand48(double, double);
point_t system_fn(params_t, point_t, double);

int main()
{
	srand(reinterpret_cast<unsigned long>(&NUM_STEPS));

	params_t params;
	point_t point;

	// Set params values
	{
		params.alpha = 1.4;
		params.beta = 0.3;
		params.gamma = 2.1;
		params.theta = 0.6;
	}

	// Print params values
	{
		std::cout
			<< "alpha = " << params.alpha << '\n'
			<< "beta = " << params.beta << '\n'
			<< "gamma = " << params.gamma << '\n'
			<< "theta = " << params.theta << '\n'
			<< '\n';
	}

	// Set initial point coordinates
	{
		point.x = 1;
		point.y = -1;
	}

	// Print initial point coordinates
	{
		std::cout
			<< "x_0 = " << point.x << '\n'
			<< "y_0 = " << point.y << '\n'
			<< '\n';
	}

	// Write data to file and terminal
	{
		std::ofstream file("data.dat");

		for (unsigned long i = 0; i < NUM_STEPS; ++i)
		{
			file << point.x << ' ' << point.y << '\n';
			point = system_fn(params, point, dt);
		}

		file.close();
	}

	return 0;
}

double drand48(const double min, const double max)
{
	return (max - min) * drand48() + min;
}

point_t system_fn(const params_t params, const point_t point, const double dt)
{
	return point_t
	{
		point.x + (params.alpha * point.x - params.beta * point.x * point.y) * dt,
		point.y + (params.theta * point.x * point.y - params.gamma * point.y) * dt
	};
}
