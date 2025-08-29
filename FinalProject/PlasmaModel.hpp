#ifndef PLASMA_MODEL_H
#define PLASMA_MODEL_H

#include <random>
#include <string>
#include <vector>

class PlasmaModel {
  private:
    int N; // N grid
    float a; // grid spacing
    int D; // dimensions of the grid
    int vecSize;


    int timeSteps; // total number of timeSteps of the integrator
    float dt; // time step of the HMC
    int L; // L steps for the leap frog integrator

    std::vector<float> q; // sampled variable (in this case, Energy)
    std::vector<float> p; // conjugate momentum
    std::vector<float> forceField; // force field of the medium
    std::vector<float> energyField; // energy field from the medium
    
    float lambda, beta; // simulation parameters
    float T; // temperature

    std::random_device device;
    std::mt19937 rng;  // random number generator
    std::normal_distribution<float> gDist;  // gaussian distribution
    std::uniform_real_distribution<float> uDist; // uniform distribution for the metropolis step 
    
    void CalculateForceField();
    void LeapFrogIntegrator();
    float CalculateHamiltonian();

  public:
    PlasmaModel(int N_p, float a_p, int D_p, int timeSteps_p, float dt_p, int L_p, float mTherm_p, float gConstant_p, float T_p);
    PlasmaModel(float lambda_p, float beta_p, int N_p, float a_p, int D_p, int timeSteps_p, float dt_p, int L_p, float T_p);

    void InitializeGrid();
    void RunSimulation();
    void ExportData(const std::string& file);
    const std::vector<float>& GetEnergyField() {return energyField;}
};

#endif //PLASMA_MODEL_H
