#ifndef PLASMA_MODEL_H
#define PLASMA_MODEL_H

#include <random>
#include <vector>

class PlasmaModel {
  private:
    int N; //N x N grid
    double a; //Grid spacing

    double dt; //time step of the HMC

    std::vector<float> q; //sampled variable (in this case, Energy)
    std::vector<float> p; //conjugate momentum

    std::random_device device;
    std::mt19937 rng;  // Random number generator
    std::normal_distribution<float> gDist;  // Gaussian distribution
    std::uniform_real_distribution<float> uDist; //Uniform distribution 

  public:
    PlasmaModel(int N_p, double a_p, double dt_p);

    void InitializeGrid();


};

#endif //PLASMA_MODEL_H
