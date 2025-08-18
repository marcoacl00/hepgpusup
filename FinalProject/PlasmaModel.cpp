#include "PlasmaModel.hpp"
#include <cmath>
#include <vector>

/* PRIVATE METHODS */

void PlasmaModel::CalculateForceField(std::vector<float> &forceField) {
  for (int i = 0; i < forceField.size(); i++) {
    forceField[i] = 2 * q[i] + 4 * lambda * (q[i] * q[i] - 1) * q[i];
    float J = 0;

    int stride = 1;
    for (int k = 0; k < D; k++) {
      int coord = (k / stride) % N;

      int forward, backward;

      if (coord == N - 1) {
        forward = k - (N - 1) * stride;
      } else {
        forward = k + stride;
      }
      if (coord == 0) {
        backward = k + (N - 1) * stride;
      } else {
        forward = k - stride;
      }

      J += forward + backward;
    }

    forceField[i] += -2 * beta * J;
    forceField[i] = -forceField[i];
  }
}

void PlasmaModel::LeapFrogIntegrator(float lambda, float beta) {
  std::vector<float> forceField = std::vector<float>(pow(N, D), 0);
  int vecSize = pow(N, D);

  for (int k = 0; k < L; k++) {
    CalculateForceField(forceField);
    for (int i = 0; i < vecSize; i++) {
      p[i] += 0.5 * dt * forceField[i];
      q[i] += dt * p[i];
    }

    CalculateForceField(forceField);
    for (int i = 0; i < pow(N, D); i++) {
      p[i] += 0.5 * dt * forceField[i];
    }
  }
}

/* PUBLIC METHODS */

PlasmaModel::PlasmaModel(int N_p, float a_p, int D_p, int timeSteps_p, float dt_p, int L_p, float lambda_p, float beta_p)
    : N(N_p), a(a_p), D(D_p), timeSteps(timeSteps_p), dt(dt_p), L(L_p),
      rng(device()), gDist(0.0, 1.0), q(pow(N, D), 0), p(pow(N, D), 0),
      lambda(lambda_p), beta(beta_p) {}

void PlasmaModel::InitializeGrid() {
  for (int i = 0; i < pow(N, D); i++) {
    q[i] = gDist(rng);
  }
}
