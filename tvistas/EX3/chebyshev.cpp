#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <vector>
#include <fstream>

using Complex = std::complex<double>;
using SparseMatrix = Eigen::SparseMatrix<double>;
using Eigen::MatrixXd;
using Eigen::VectorXcd;

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

  SparseMatrix Hamiltonian;
  VectorXcd currentSolution;

  std::vector<double> positionExpValues;

  /* MEHTODS */

  double GenerateChebyshevCoeff(double a, int n) {
    double r = (n == 0) ? 1.0 : 2.0;
    return r * std::cyl_bessel_j(n, a);
  }

  
  void BuildHamiltonian() {
    std::vector<Eigen::Triplet<double>> triplets;
    double kineticCoeff = -hbar * hbar / (2.0 * dx * dx);

    for (int i = 0; i < numStepsX; ++i) {
      double x = xmin + i * dx;
      double Vx = Potential(x);

      if (i == 0) {
        // Left boundary: one-sided second derivative
        triplets.emplace_back(0, 0, kineticCoeff * 2.0 + Vx);
        triplets.emplace_back(0, 1, kineticCoeff * -5.0);
        triplets.emplace_back(0, 2, kineticCoeff * 4.0);
        triplets.emplace_back(0, 3, kineticCoeff * -1.0);
      } else if (i == numStepsX - 1) {
        // Right boundary: one-sided second derivative
        triplets.emplace_back(i, i, kineticCoeff * 2.0 + Vx);
        triplets.emplace_back(i, i - 1, kineticCoeff * -5.0);
        triplets.emplace_back(i, i - 2, kineticCoeff * 4.0);
        triplets.emplace_back(i, i - 3, kineticCoeff * -1.0);
      } else {
        // Interior: standard 3-point stencil
        triplets.emplace_back(i, i, kineticCoeff * -2.0 + Vx);
        triplets.emplace_back(i, i - 1, kineticCoeff);
        triplets.emplace_back(i, i + 1, kineticCoeff);
      }
    }

    Hamiltonian.resize(numStepsX, numStepsX);
    Hamiltonian.setFromTriplets(triplets.begin(), triplets.end());
  }

   
  double CalculateEmin() {
    double result = Potential(xmin);
    for (int i = 1; i < numStepsX; i++) {
      double potential = Potential(xmin + i*dx); 
      if (potential < result) {
        result = potential;
      }
    }
    return result;
  }

  double CalculateEmax() {
    double result = Potential(xmin);
    for (int i = 1; i < numStepsX; i++) {
      double potential = Potential(xmin + i*dx); 
      if (potential > result) {
        result = potential;
      }
    }
    return result + hbar*hbar*M_PI*M_PI/(2*dx*dx);
  }

  void NormalizeHamiltonian() {
    SparseMatrix I(numStepsX, numStepsX);
    I.setIdentity();
    double Emin = CalculateEmin();
    double Emax = CalculateEmax();
    double dE = Emax-Emin;
    Hamiltonian = 2*(Hamiltonian - (dE/2 + Emin)*I)/dE;
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
    Normalize();
    
    double result = 0;
    for (int i = 0; i < numStepsX; i++) {
      double x = xmin + i*dx;
      result += x * std::norm(currentSolution[i]); 
    }

    return dx*result;
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
      currentSolution[i] = f(xmin + i*dx);
    }
  }

  void SolveSystem(int M) {
    BuildHamiltonian();
    NormalizeHamiltonian();
    
    double Emin = CalculateEmin(); double Emax = CalculateEmax();
    double dE = Emax - Emin;

    VectorXcd prevChebyshevTerm(numStepsX);
    VectorXcd prevPrevChebyshevTerm(numStepsX);
    VectorXcd currentChebyshevTerm(numStepsX);

    positionExpValues[0] = CalculatePosExpectedValue();

    for (int i = 1; i < numStepsT; i++) {
      double a = dE*i*dt/(2*hbar);
      VectorXcd newSolution(numStepsX);
      
      currentChebyshevTerm = currentSolution;
      newSolution += GenerateChebyshevCoeff(a, 0) * currentChebyshevTerm;
      prevChebyshevTerm = currentChebyshevTerm;
      
      currentChebyshevTerm = Complex(0,-1)*Hamiltonian*prevChebyshevTerm;
      newSolution += GenerateChebyshevCoeff(a, 1) * currentChebyshevTerm;
      prevPrevChebyshevTerm = prevChebyshevTerm;
      prevChebyshevTerm = currentChebyshevTerm;
      
      for (int k = 2; k < M+1; k++) {
        currentChebyshevTerm = Complex(0,-1)*2.0*Hamiltonian*prevChebyshevTerm + prevPrevChebyshevTerm;
        newSolution += GenerateChebyshevCoeff(a, k)*currentChebyshevTerm;
        prevPrevChebyshevTerm = prevChebyshevTerm;
        prevChebyshevTerm = currentChebyshevTerm;
      }

      currentSolution = std::exp(Complex(0, -1)*(Emin*i*dt/hbar + a))*newSolution;
      positionExpValues[i] = CalculatePosExpectedValue();

      if (i % 100 == 0) {
        std::cout << "Progress: " << i * dt << " / " << tmax << std::endl;
      }
    }
  }

  void PrintResults() {
    for (int t = 0; t < numStepsT; t+=10) {
      std::cout << positionExpValues[t] << std::endl; 
    }
  }

  void SaveResults(const std::string& filename) {
    std::ofstream outfile(filename);
    for (int t = 0; t < numStepsT; ++t) {
      outfile << t * dt << " " << positionExpValues[t] << std::endl;
    } 
    outfile.close();
  }
};

double LennardJonesPotential(double x) {
  double sigma = 2.099518748076819;
  double epsilon = 4.642973059984742;

  if (x < 1e-10)
    x = 1e-10; // Avoid division by zero
  double sr6 = std::pow(sigma / x, 6);
  return 4.0 * epsilon * (sr6 * sr6 - sr6);

}

double HOPotential(double x) {
  double w2 = 400;
  return 0.5 * w2 * x * x;
}

Complex GaussianPulse(double x) {
  double x0 = 2.55;
  double k = 0;
  double amplitude = pow(1/M_PI, 0.25);
  double sigma = 1;

  double real_part = amplitude * std::exp(-std::pow(x - x0, 2)*20 / (2 * sigma * sigma));
  Complex phase = std::exp(Complex(0, k * x));  

  return real_part * phase;
  
}

int main (int argc, char *argv[]) {
  double dx = 0.005; double dt = 0.01;
  double tmin = 0; double tmax = 10;
  double xmin = 1.5; double xmax = 6.5;
  double hbar = 1;

  SchrodingerSolver solver(dx, dt, tmin, tmax, xmin, xmax, hbar, LennardJonesPotential);

  solver.SetInitialState(GaussianPulse);

  solver.SolveSystem(50); 
  
  solver.SaveResults("data.txt");
  //solver.PrintResults();
  
  return 0;
}
