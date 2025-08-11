#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Lattice = std::vector<std::vector<double>>;

class HMC {
private:
  // Simulation parameters
  int N;     // Grid size
  double dt; // Time step
  double H;  // External field
  int L;     // Leapfrog steps per trajectory

  // Internal state
  Lattice q, p;
  std::random_device device;
  std::vector<double> avgQ, avgP;

  Lattice CalculateForce(double T, double lambda) {
    Lattice force(N, std::vector<double>(N, 0));
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int ip = (i + 1) % N, im = (i - 1 + N) % N;
        int jp = (j + 1) % N, jm = (j - 1 + N) % N;
        force[i][j] = -(q[i][j] - (q[i][jp] + q[ip][j] + q[i][jm] + q[im][j]) +
                        T * q[i][j] + 2 * lambda * pow(q[i][j], 3) + H);
      }
    }
    return force;
  }

  void LeapFrog(double T, double lambda) {
    Lattice force = CalculateForce(T, lambda);
    for (int k = 0; k < L; k++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          p[i][j] += 0.5 * dt * force[i][j];
          q[i][j] += dt * p[i][j];
        }
      }
      force = CalculateForce(T, lambda);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          p[i][j] += 0.5 * dt * force[i][j];
        }
      }
    }
  }

  double Hamiltonian(double T, double lambda) {
    double kin = 0, pot = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        kin += 0.5 * p[i][j] * p[i][j];
        int ip = (i + 1) % N, jp = (j + 1) % N;
        pot += 0.5 * (pow(q[i][j] - q[i][jp], 2) + pow(q[i][j] - q[ip][j], 2)) +
               T * q[i][j] * q[i][j] + lambda * pow(q[i][j], 4) + H * q[i][j];
      }
    }
    return kin + pot;
  }

  void UpdateLattice(double T, double lambda) {
    std::mt19937 gen(device());
    std::normal_distribution<double> gauss(0, 1);
    std::uniform_real_distribution<double> uniform(0, 1);

    for (auto &row : p)
      for (double &val : row)
        val = gauss(gen);

    double H_old = Hamiltonian(T, lambda);
    Lattice q_old = q;
    LeapFrog(T, lambda);
    double H_new = Hamiltonian(T, lambda);

    double dH = H_new - H_old;
    if (uniform(gen) > exp(-dH))
      q = q_old;
  }

  double AvgQ() const {
    double sum = 0;
    for (const auto &row : q)
      sum += std::accumulate(row.begin(), row.end(), 0.0);
    return sum / (N * N);
  }

public:
  HMC(int N_p = 16, double dt_p = 0.01, double H_p = 0.0, int L_p = 10)
      : N(N_p), dt(dt_p), H(H_p), L(L_p), q(N, std::vector<double>(N)),
        p(N, std::vector<double>(N)) {}

  void InitializeLattice() {
    std::mt19937 gen(device());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &row : q)
      for (double &val : row)
        val = dist(gen);
  }

  void RunSimulation(int steps, double T, double lambda) {
    avgQ.resize(steps);
    for (int i = 0; i < steps; i++) {
      UpdateLattice(T, lambda);
      avgQ[i] = AvgQ();
    }
  }

  double FindCriticalTemperature(double Tc_guess, double T_min, double T_max,
                                 double dT, int thermal_steps,
                                 int measure_steps) {
    double Tc_current = Tc_guess;
    double Tc_prev = 0.0;
    const double tolerance = 0.001;
    int iteration = 0;

    while (std::abs(Tc_current - Tc_prev) > tolerance && iteration < 20) {
      Tc_prev = Tc_current;
      double peak_T = 0.0;
      double max_chi = 0.0;

      std::ofstream out("Tc_search_iter_" + std::to_string(iteration) + ".txt");
      out << "# T \t <|m|> \t chi\n";

      for (double T = T_min; T <= T_max; T += dT) {
        std::cout << "Progress: T = " << T << std::endl;
        double lambda = (T - Tc_current) / T;
        InitializeLattice();
        RunSimulation(thermal_steps, T, lambda); // Thermalize
        RunSimulation(measure_steps, T, lambda); // Measure

        double avg_m = 0, avg_m2 = 0;
        for (size_t i = avgQ.size() / 2; i < avgQ.size(); i++) {
          avg_m += std::abs(avgQ[i]);
          avg_m2 += avgQ[i] * avgQ[i];
        }
        avg_m /= (avgQ.size() / 2);
        avg_m2 /= (avgQ.size() / 2);
        double chi = N * N * (avg_m2 - avg_m * avg_m) / T;

        out << T << "\t" << avg_m << "\t" << chi << "\n";

        if (chi > max_chi) {
          max_chi = chi;
          peak_T = T;
        }
      }
      out.close();
      Tc_current = peak_T;
      std::cout << "Iteration " << iteration << ": Tc = " << Tc_current << "\n";
      iteration++;
    }
    return Tc_current;
  }
};

void show_help(const char *program_name) {
  std::cout << "Usage: " << program_name << " [options]\n"
            << "Options:\n"
            << "  -N <size>        Lattice size (default: 16)\n"
            << "  -dt <step>       Time step (default: 0.01)\n"
            << "  -H <field>       External field (default: 0.0)\n"
            << "  -L <steps>       Leapfrog steps (default: 10)\n"
            << "  -Tc <guess>      Initial Tc guess (default: 2.3)\n"
            << "  -Tmin <value>    Minimum T for scan (default: 1.5)\n"
            << "  -Tmax <value>    Maximum T for scan (default: 3.0)\n"
            << "  -dT <step>       Temperature step (default: 0.05)\n"
            << "  -therm <steps>   Thermalization steps (default: 1000)\n"
            << "  -meas <steps>    Measurement steps (default: 2000)\n";
}

int main(int argc, char **argv) {
  // Default parameters
  int N = 16;
  double dt = 0.01;
  double H = 0.0;
  int L = 10;
  double Tc_guess = 2.3;
  double T_min = 1.5, T_max = 3.0, dT = 0.05;
  int thermal_steps = 1000, measure_steps = 2000;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-N" && i + 1 < argc)
      N = atoi(argv[++i]);
    else if (arg == "-dt" && i + 1 < argc)
      dt = atof(argv[++i]);
    else if (arg == "-H" && i + 1 < argc)
      H = atof(argv[++i]);
    else if (arg == "-L" && i + 1 < argc)
      L = atoi(argv[++i]);
    else if (arg == "-Tc" && i + 1 < argc)
      Tc_guess = atof(argv[++i]);
    else if (arg == "-Tmin" && i + 1 < argc)
      T_min = atof(argv[++i]);
    else if (arg == "-Tmax" && i + 1 < argc)
      T_max = atof(argv[++i]);
    else if (arg == "-dT" && i + 1 < argc)
      dT = atof(argv[++i]);
    else if (arg == "-therm" && i + 1 < argc)
      thermal_steps = atoi(argv[++i]);
    else if (arg == "-meas" && i + 1 < argc)
      measure_steps = atoi(argv[++i]);
    else if (arg == "-h" || arg == "--help") {
      show_help(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      show_help(argv[0]);
      return 1;
    }
  }

  // Run simulation
  HMC hmc(N, dt, H, L);
  double Tc = hmc.FindCriticalTemperature(Tc_guess, T_min, T_max, dT,
                                          thermal_steps, measure_steps);
  std::cout << "\nFinal critical temperature: Tc = " << Tc << "\n";

  return 0;
}
