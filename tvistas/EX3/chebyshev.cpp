#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <vector>

using Complex = std::complex<double>;
using MatrixXd = Eigen::MatrixXd;
using MatrixXcd = Eigen::MatrixXcd;
using SparseXd = Eigen::SparseMatrix<double>;
using Eigen::VectorXcd;
using Eigen::VectorXd;

class SchrodingerSolver {
private:
  /* PARAMETERS */

  int numStepsX;
  int numStepsT;

  double dx, dt;

  double tmin, tmax, xmin, xmax;

  double hbar;

  std::function<double(double)> Potential;
  std::vector<double> chebyshevCoeff;

  SparseXd Hamiltonian;
  MatrixXcd ChebyshevProp;
  VectorXcd currentSolution;

  std::vector<double> positionExpValues;

  /* MEHTODS */

  Complex GenerateChebyshevCoeff(int n, double lambda) {
    double r = (n == 0) ? 1.0 : 2.0;
    return r * std::pow(Complex(0, -1), n) * std::cyl_bessel_j(n, dt * lambda);
  }

  void BuildHamiltonian() {
    std::vector<Eigen::Triplet<double>> triplets;
    double kineticDiag = 2.0 / (dx * dx) * (1.0 / (2.0));
    double kineticOffDiag = -1.0 / (dx * dx) * (1.0 / (2.0));

    triplets.reserve(3 * numStepsX - 2);

    for (int i = 0; i < numStepsX; ++i) {
      double x = xmin + i * dx;
      double Vx = Potential(x);
      triplets.emplace_back(i, i, kineticDiag + Vx);

      if (i > 0) {
        triplets.emplace_back(i, i - 1, kineticOffDiag);
      }

      if (i < numStepsX - 1) {
        triplets.emplace_back(i, i + 1, kineticOffDiag);
      }
    }

    Hamiltonian.resize(numStepsX, numStepsX);
    Hamiltonian.setFromTriplets(triplets.begin(), triplets.end());
    Hamiltonian.makeCompressed();
  }

  double NormalizeHamiltonian() {
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(Hamiltonian);
    VectorXd ev = solver.eigenvalues();
    double minev = ev.minCoeff();
    double maxev = ev.maxCoeff();
    double lambda = maxev - minev;
    Hamiltonian = (1 / lambda) * Hamiltonian;

    return lambda;
  }

  void BuildChebyshevMatrix(double lambda, int M) {
    MatrixXd A0 = MatrixXd::Identity(numStepsX, numStepsX);
    MatrixXd A1 = Hamiltonian;

    ChebyshevProp = GenerateChebyshevCoeff(0, lambda) * A0 +
                    GenerateChebyshevCoeff(1, lambda) * A1;

    for (int k = 2; k < M + 1; k++) {
      MatrixXd A2 = 2.0 * Hamiltonian * A1 - A0;
      ChebyshevProp += GenerateChebyshevCoeff(k, lambda) * A2;
      A0 = A1;
      A1 = A2;
    }
  }

  void Normalize() {
    double norm = 0.0;
    for (const auto &val : currentSolution) {
      norm += std::norm(val);
    }
    norm = std::sqrt(norm * dx);
    for (auto &val : currentSolution) {
      val /= norm;
    }
  }

  double CalculatePosExpectedValue() {
    double result = 0;
    for (int i = 0; i < numStepsX; i++) {
      double x = xmin + i * dx;
      result += x * std::norm(currentSolution[i]);
    }

    return dx * result;
  }

public:
  SchrodingerSolver(double dx_p, double dt_p, double tmin_p, double tmax_p,
                    double xmin_p, double xmax_p, double hbar_p,
                    std::function<double(double)> potential_p)
      : dx(dx_p), dt(dt_p), tmin(tmin_p), tmax(tmax_p), xmin(xmin_p),
        xmax(xmax_p), hbar(hbar_p), Potential(potential_p) {
    numStepsX = (xmax - xmin) / dx;
    numStepsT = (tmax - tmin) / dt;

    positionExpValues.resize(numStepsT);
  }

  void SetInitialState(std::function<Complex(double)> f) {
    currentSolution = VectorXcd::Zero(numStepsX);
    for (int i = 0; i < numStepsX; i++) {
      currentSolution[i] = f(xmin + i * dx);
    }
  }

  void SolveSystem(int M) {
    BuildHamiltonian();
    double lambda = NormalizeHamiltonian();

    positionExpValues[0] = CalculatePosExpectedValue();

    BuildChebyshevMatrix(lambda, M);

    for (int i = 1; i < numStepsT; i++) {
      VectorXcd newSolution = ChebyshevProp * currentSolution;
      currentSolution = newSolution;
      positionExpValues[i] = CalculatePosExpectedValue();

      if (i % 100 == 0) {
        std::cout << "Progress: " << i * dt << " / " << tmax << std::endl;
      }
    }
  }

  void PrintResults() {
    for (int t = 0; t < numStepsT; t += 10) {
      std::cout << positionExpValues[t] << std::endl;
    }
  }

  void SaveResults(const std::string &filename) {
    std::ofstream outfile(filename);
    for (int t = 0; t < numStepsT; t++) {
      outfile << t * dt << " " << positionExpValues[t] << std::endl;
    }
    outfile.close();
  }
};

double LennardJonesPotential(double x) {
  double sigma = 2.099518748076819;
  double epsilon = 4.642973059984742;

  if (x < 1e-10)
    x = 1e-10;
  double sr6 = std::pow(sigma / x, 6);
  return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

double HOPotential(double x) {
  double w2 = 1;
  return 0.5 * w2 * x * x;
}

Complex GaussianPulse(double x) {
  double x0 = 0;
  double k = 2;
  double sigma = 1;
  double amplitude = pow(1 / (M_PI * sigma), 0.25);

  return std::exp(-((x - x0) * (x - x0 - Complex(0, 2) * k * sigma)) /
                  (2 * sigma)) /
         (pow(M_PI, 0.25) * pow(sigma, 0.25));
}

int main(int argc, char *argv[]) {
  double dx = 0.07874;
  double dt = 0.05;
  double tmin = 0;
  double tmax = 100;
  double xmin = -5;
  double xmax = 5;
  double hbar = 1;

  std::ofstream initialPulseFile("initial_pulse.txt");
  for (double x = xmin; x <= xmax; x += dx) {
    Complex psi = GaussianPulse(x);
    initialPulseFile << x << " " << psi.real() << " " << psi.imag() << " "
                     << std::norm(psi) << std::endl;
  }
  initialPulseFile.close();

  SchrodingerSolver solver(dx, dt, tmin, tmax, xmin, xmax, hbar, HOPotential);

  solver.SetInitialState(GaussianPulse);

  solver.SolveSystem(40);

  solver.SaveResults("data.txt");

  return 0;
}
