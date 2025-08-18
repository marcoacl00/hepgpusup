#ifndef PLASMA_MODEL_H
#define PLASMA_MODEL_H

#include <random>
#include <vector>

class PlasmaModel {
  private:
    int N; // N grid
    float a; // grid spacing
    int D; // dimensions of the grid


    int timeSteps; // total number of timeSteps of the integrator
    float dt; // time step of the HMC
    int L; // L steps for the leap frog integrator

    std::vector<float> q; // sampled variable (in this case, Energy)
    std::vector<float> p; // conjugate momentum
    
    float lambda, beta; // simulation parameters

    std::random_device device;
    std::mt19937 rng;  // random number generator
    std::normal_distribution<float> gDist;  // gaussian distribution
    
    void CalculateForceField(std::vector<float>& forceField);
    void LeapFrogIntegrator(float lambda, float beta);

  public:
    PlasmaModel(int N_p, float a_p, int D_p, int timeSteps_p, float dt_p, int L_p, float lambda_p, float beta_p);

    void InitializeGrid();


};

#endif //PLASMA_MODEL_H
