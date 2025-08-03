#include <fstream>
#include <cmath>
#include <iomanip>

using type_t = double;

constexpr const type_t
	H_MIN = 1E-16,
	H_MAX = 1E-1,
	H_STEP = 10;

type_t f(type_t);
type_t df(type_t);
type_t forward_diff(type_t (*f)(type_t), type_t, type_t);
type_t central_diff(type_t (*f)(type_t), type_t, type_t);

int main()
{
	const type_t x0 = 1;
	std::ofstream file;

	// Forward Difference
	{
		file.open("forward_diff.dat");

		for (type_t h = H_MIN; h <= H_MAX; h *= H_STEP)
			file << h << ' ' << std::fabs(df(x0) - forward_diff(f, x0, h)) << '\n';

		file.close();
	}

	// Central Difference
	{
		file.open("central_diff.dat");

		for (type_t h = H_MIN; h <= H_MAX; h *= H_STEP)
			file << h << ' ' << std::fabs(df(x0) - central_diff(f, x0, h)) << '\n';

		file.close();
	}

	return 0;
}

type_t f(const type_t x)
{
	return std::sin(x);
}

type_t df(const type_t x)
{
	return std::cos(x);
}

type_t forward_diff(type_t (*f)(type_t), const type_t x, const type_t h)
{
	return (f(x + h) - f(x)) / h;
}

type_t central_diff(type_t (*f)(type_t), const type_t x, const type_t h)
{
	return (f(x + h) - f(x - h)) / (2 * h);
}
