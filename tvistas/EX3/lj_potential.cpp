#include "chebyshev_solver.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // Test parameters
    const size_t N = 3;  // Small size for easy verification
    const double tolerance = 1e-10;
    
    // Create and build the differentiation matrix
    ChebyshevSolver solver;
    solver.BuildDiffMatrix(N);
    
    // Get the results
    const Eigen::MatrixXd& D = solver.GetDiffMatrix();
    const std::vector<double>& nodes = solver.GetNodalPoints();
    
    // Print the nodal points
    std::cout << "Chebyshev nodes (N = " << N << "):\n";
    for (size_t i = 0; i < nodes.size(); ++i) {
        std::cout << "x[" << i << "] = " << std::setprecision(12) << nodes[i] << "\n";
    }
    std::cout << "\n";
    
    // Print the differentiation matrix
    std::cout << "Differentiation matrix:\n" << D << "\n\n";
    
    // Verification tests
    
    // Test 1: Check that the sum of each row is approximately zero
    bool row_sum_test_passed = true;
    for (int i = 0; i < D.rows(); ++i) {
        double row_sum = D.row(i).sum();
        if (std::abs(row_sum) > tolerance) {
            std::cerr << "Row sum test failed for row " << i 
                      << " (sum = " << row_sum << ")\n";
            row_sum_test_passed = false;
        }
    }
    
    if (row_sum_test_passed) {
        std::cout << "✓ Row sum test passed (all rows sum to nearly zero)\n";
    }
    
    // Test 2: Check known values for N=5 case
    if (N == 5) {
        bool known_values_test_passed = true;
        
        // Check diagonal elements (approximate values)
        const double expected_diag[] = {8.5, -2.61803, 1.17082, -1.17082, 2.61803};
        
        for (int i = 0; i < 5; ++i) {
            if (std::abs(D(i,i) - expected_diag[i]) > 1e-5) {
                std::cerr << "Diagonal value mismatch at (" << i << "," << i 
                          << "): expected " << expected_diag[i] 
                          << ", got " << D(i,i) << "\n";
                known_values_test_passed = false;
            }
        }
        
        if (known_values_test_passed) {
            std::cout << "✓ Known values test passed for N=5 case\n";
        }
    }
    
    // Test 3: Check anti-symmetry property (D_ij ≈ -D_ji for i≠j)
    bool antisymmetry_test_passed = true;
    for (int i = 0; i < D.rows(); ++i) {
        for (int j = 0; j < D.cols(); ++j) {
            if (i != j && std::abs(D(i,j) + D(j,i)) > tolerance) {
                std::cerr << "Anti-symmetry test failed at (" << i << "," << j 
                          << "): Dij = " << D(i,j) << ", Dji = " << D(j,i) << "\n";
                antisymmetry_test_passed = false;
            }
        }
    }
    
    if (antisymmetry_test_passed) {
        std::cout << "✓ Anti-symmetry test passed\n";
    }
    
    return 0;
}
