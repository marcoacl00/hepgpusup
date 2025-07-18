#include "chebyshev_solver.hpp"
#include <cmath>

void ChebyshevSolver::BuildDiffMatrix(size_t N) {
    nodalPoints.clear();
    
    size_t numPoints = N + 1;
    nodalPoints.resize(numPoints);
    DiffMatrix.resize(numPoints, numPoints);
    
    for (size_t j = 0; j < numPoints; ++j) {
        nodalPoints[j] = std::cos(M_PI * j / N);
    }
    
    for (size_t i = 0; i < numPoints; ++i) {
        for (size_t j = 0; j < numPoints; ++j) {
            if (i == j) {
                if (i == 0) {
                    DiffMatrix(i, j) = (2.0 * N * N + 1.0) / 6.0;
                } else if (i == N) {
                    DiffMatrix(i, j) = -(2.0 * N * N + 1.0) / 6.0;
                } else {
                    DiffMatrix(i, j) = -nodalPoints[i] / (2.0 * (1.0 - nodalPoints[i] * nodalPoints[i]));
                }
            } else {
                double ci = (i == 0 || i == N) ? 2.0 : 1.0;
                double cj = (j == 0 || j == N) ? 2.0 : 1.0;
                
                DiffMatrix(i, j) = (ci / cj) * std::pow(-1.0, i + j) / (nodalPoints[i] - nodalPoints[j]);
            }
        }
    }
}

Eigen::VectorXd ChebyshevSolver::BuildForceVector
