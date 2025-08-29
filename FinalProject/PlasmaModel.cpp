#include "PlasmaModel.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

/* PRIVATE METHODS */

void PlasmaModel::CalculateForceField() {
  for (int i = 0; i < forceField.size(); i++) {
    float J = 0.0f;
    int stride = 1;
    for (int mu = 0; mu < D; ++mu) {
      int coord = (i / stride) % N;
      int forward = (coord == N - 1) ? (i - (N - 1) * stride) : (i + stride);
      int backward = (coord == 0) ? (i + (N - 1) * stride) : (i - stride);
      J += q[forward] + q[backward];
      stride *= N;
    }

    forceField[i] =
        2 * beta * J - 2 * q[i] - 4 * lambda * (q[i] * q[i] - 1) * q[i];
  }
}

void PlasmaModel::LeapFrogIntegrator() {
  for (int k = 0; k < L; k++) {
    CalculateForceField();
    for (int i = 0; i < vecSize; i++) {
      p[i] += 0.5 * dt * forceField[i];
      q[i] += dt * p[i];
    }

    CalculateForceField();
    for (int i = 0; i < vecSize; i++) {
      p[i] += 0.5 * dt * forceField[i];
    }
  }
}

float PlasmaModel::CalculateHamiltonian() {
  float kEnergy = 0;
  float action = 0;
  for (int i = 0; i < vecSize; i++) {
    kEnergy += 0.5 * p[i] * p[i];

    action += q[i] * q[i] + lambda * (q[i] * q[i] - 1) * (q[i] * q[i] - 1);
    int stride = 1;
    for (int k = 0; k < D; k++) {
      int coord = (i / stride) % N;
      int neighbour = (coord == N - 1) ? (i - (N - 1) * stride) : (i + stride);

      action += -2 * beta * q[i] * q[neighbour];
      stride *= N;
    }
  }

  return kEnergy + action;
}

/* PUBLIC METHODS */

PlasmaModel::PlasmaModel(int N_p, float a_p, int D_p, int timeSteps_p,
                         float dt_p, int L_p, float mTherm_p, float gConstant_p,
                         float T_p)
    : N(N_p), a(a_p), D(D_p), vecSize(pow(N, D)), timeSteps(timeSteps_p),
      dt(dt_p), L(L_p), T(T_p), rng(device()), gDist(0.0, 1.0),
      uDist(0.0, 1.0) {
  float c = gConstant_p * pow(a_p, 4 - D_p) / 24.0;
  float k = mTherm_p * mTherm_p * a_p * a_p + 2 * D_p;
  beta = (k - sqrt(k * k + 32 * c)) / (-8 * c);
  lambda = c * beta * beta;
  q = std::vector<float>(vecSize, 0);
  p = std::vector<float>(vecSize, 0);
  forceField = std::vector<float>(vecSize, 0);
  energyField = std::vector<float>(vecSize, 0);
}
PlasmaModel::PlasmaModel(float lambda_p, float beta_p, int N_p, float a_p,
                         int D_p, int timeSteps_p, float dt_p, int L_p,
                         float T_p)
    : N(N_p), a(a_p), D(D_p), vecSize(pow(N, D)), timeSteps(timeSteps_p),
      dt(dt_p), L(L_p), T(T_p), rng(device()), gDist(0.0, 1.0), uDist(0.0, 1.0),
      lambda(lambda_p), beta(beta_p) {
  q = std::vector<float>(vecSize, 0);
  p = std::vector<float>(vecSize, 0);
  forceField = std::vector<float>(vecSize, 0);
  energyField = std::vector<float>(vecSize, 0);
}

void PlasmaModel::InitializeGrid() {
  for (int i = 0; i < vecSize; i++) {
    q[i] = gDist(rng);
  }
}

void PlasmaModel::RunSimulation() {
  gDist = std::normal_distribution<float>(0.0, sqrt(T));
  for (int t = 0; t < timeSteps; t++) {
    for (int i = 0; i < vecSize; i++) {
      p[i] = gDist(rng);
    }

    float initH = CalculateHamiltonian();
    std::vector<float> qOld = q;
    LeapFrogIntegrator();
    float finalH = CalculateHamiltonian();

    float dH = finalH - initH;
    float acceptanceProb = std::exp(-dH);
    if (acceptanceProb > 1.0f)
      acceptanceProb = 1.0f;

    if (uDist(rng) > acceptanceProb) {
      q = qOld;
    }

    if (t % std::max(1, timeSteps / 100) == 0) {
      float progress = 100.0f * t / timeSteps;
      std::cout << "Acc rate: " << acceptanceProb << "; Step: " << t*100/timeSteps << "/100" << std::endl;
    }
  }

  for (int i = 0; i < vecSize; i++) {
    float action = q[i] * q[i] + lambda * (q[i] * q[i] - 1) * (q[i] * q[i] - 1);
    int stride = 1;
    for (int k = 0; k < D; k++) {
      int coord = (i / stride) % N;
      int neighbour = (coord == N - 1) ? (i - (N - 1) * stride) : (i + stride);

      action += -2 * beta * q[i] * q[neighbour];
      stride *= N;
    }
    energyField[i] = action;
  }
}

void PlasmaModel::ExportData(const std::string &file) {
  std::ofstream out(file);
  if (D == 1) {
    for (int i = 0; i < N; i++) {
      out << energyField[i];
      if (i < N - 1)
        out << ",";
    }
    out << "\n";
  } else if (D == 2) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int idx = i * N + j;
        out << energyField[idx];
        if (j < N - 1)
          out << ",";
      }
      out << "\n";
    }
  } else {
    for (int idx = 0; idx < vecSize; idx++) {
      int tmp = idx;
      std::vector<int> coords(D);
      for (int d = 0; d < D; d++) {
        coords[d] = tmp % N;
        tmp /= N;
      }
      for (int d = 0; d < D; d++) {
        out << coords[d];
        if (d < D - 1)
          out << ",";
      }
      out << "," << energyField[idx] << "\n";
    }
  }
}
