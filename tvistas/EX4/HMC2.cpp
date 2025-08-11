#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using Lattice = std::vector<std::vector<double>>;

class HMC {
private:
  int N;
  Lattice phi, p;
  double dt;
  std::random_device device;

  Lattice CalculateForce(double T, double lambda, double H) {
    Lattice f(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int ip = (i + 1) % N, im = (i - 1 + N) % N;
        int jp = (j + 1) % N, jm = (j - 1 + N) % N;
        double qi = std::tanh(phi[i][j]);
        double qip = std::tanh(phi[ip][j]);
        double qim = std::tanh(phi[im][j]);
        double qjp = std::tanh(phi[i][jp]);
        double qjm = std::tanh(phi[i][jm]);
        double neighbor_contrib = 4.0 * qi - (qip + qim + qjp + qjm);
        double dS_dq = neighbor_contrib + 2.0 * T * qi + 4.0 * lambda * std::pow(qi, 3) + H;
        double dq_dphi = 1.0 - qi * qi;
        f[i][j] = -dS_dq * dq_dphi;
      }
    }
    return f;
  }

  void LeapFrogIntegrator(int L, double T, double lambda, double H) {
    Lattice f = CalculateForce(T, lambda, H);
    for (int k = 0; k < L; k++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          p[i][j] += 0.5 * dt * f[i][j];
          phi[i][j] += dt * p[i][j];
        }
      }
      f = CalculateForce(T, lambda, H);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          p[i][j] += 0.5 * dt * f[i][j];
        }
      }
    }
  }

  double CalculateHamiltonian(double T, double lambda, double H) {
    double kE = 0.0, pE = 0.0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        double qi = std::tanh(phi[i][j]);
        kE += 0.5 * p[i][j] * p[i][j];
        int ip = (i + 1) % N, jp = (j + 1) % N;
        double qip = std::tanh(phi[ip][j]);
        double qjp = std::tanh(phi[i][jp]);
        pE += 0.5 * (qi - qip) * (qi - qip);
        pE += 0.5 * (qi - qjp) * (qi - qjp);
        pE += T * qi * qi;
        pE += lambda * qi * qi * qi * qi;
        pE += H * qi;
      }
    }
    return kE + pE;
  }

  void UpdateLattice(int L, double T, double lambda, double H) {
    std::mt19937 gen(device());
    std::normal_distribution<double> nd(0, 1);
    std::uniform_real_distribution<double> ud(0, 1);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) p[i][j] = nd(gen);
    double H0 = CalculateHamiltonian(T, lambda, H);
    Lattice phi_old = phi;
    LeapFrogIntegrator(L, T, lambda, H);
    double H1 = CalculateHamiltonian(T, lambda, H);
    double prob = exp(-H1 + H0);
    if (ud(gen) > (prob > 1 ? 1 : prob)) phi = phi_old;
    else for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) p[i][j] = -p[i][j];
  }

  double AvgQ() {
    double s = 0.0;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) s += std::tanh(phi[i][j]);
    return s / (N * N);
  }

public:
  HMC(int Np, double dtp) : N(Np), dt(dtp) {
    phi = Lattice(N, std::vector<double>(N, 0));
    p = Lattice(N, std::vector<double>(N, 0));
  }

  void InitializeLattice() {
    std::mt19937 gen(device());
    std::uniform_real_distribution<double> ud(-1, 1);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) phi[i][j] = atanh(ud(gen));
  }

  void MeasureAtT(double T, double lambda, double H, int simSteps, int LeapSteps, double &m_abs, double &sus, double &binder) {
    InitializeLattice();
    int eqSteps = simSteps / 3;
    double m_sum = 0, m2_sum = 0, m4_sum = 0;
    for (int i = 0; i < simSteps; i++) {
      UpdateLattice(LeapSteps, T, lambda, H);
      if (i >= eqSteps) {
        double m = AvgQ();
        m_sum += std::abs(m);
        m2_sum += m * m;
        m4_sum += m * m * m * m;
      }
    }
    int measCount = simSteps - eqSteps;
    m_abs = m_sum / measCount;
    sus = N * N * (m2_sum / measCount - (m_sum / measCount) * (m_sum / measCount)) / T;
    binder = 1.0 - (m4_sum / measCount) / (3.0 * (m2_sum / measCount) * (m2_sum / measCount));
  }
};

int main() {
  int N = 16, simSteps = 10000, LeapSteps = 20;
  double dt = 0.01, H = 0.0, lambda = 1.0;
  HMC hmc(N, dt);
  std::ofstream out("Tc.txt");
  out << "Temp\tm_abs\tsus\tbinder\n";
  for (double T = 0.5; T <= 4.0; T += 0.1) {
    double m_abs, sus, binder;
    hmc.MeasureAtT(T, lambda, H, simSteps, LeapSteps, m_abs, sus, binder);
    out << T << "\t" << m_abs << "\t" << sus << "\t" << binder << "\n";
    std::cout << T << " done\n";
  }
  out.close();
  return 0;
}

