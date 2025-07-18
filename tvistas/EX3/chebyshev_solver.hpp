#ifndef CHEBYSHEV_SOLVER_H
#define CHEBYSHEV_SOLVER_H

#include "Eigen/Dense"
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class ChebyshevSolver {
  public:
    void BuildDiffMatrix(size_t N);
    const Eigen::MatrixXd& GetDiffMatrix() const { return DiffMatrix; }
    const Eigen::VectorXd& GetSoultion() const { return SolutionVector; }
    const std::vector<double>& GetNodalPoints() const { return nodalPoints; }
    
  private:
    std::function<double(double)> force;
    MatrixXd DiffMatrix;
    VectorXd SolutionVector;
    std::vector<double> nodalPoints;

    Eigen::VectorXd BuildForceVector(size_t N);
};

#endif  // CHEBYSHEV_SOLVER_H
