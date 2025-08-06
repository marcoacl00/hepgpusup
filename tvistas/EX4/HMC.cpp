#include <cmath>
#include <iostream>
#include <fstream>
#include <ostream>
#include <random>
#include <vector>

using Lattice = std::vector<std::vector<double>>;

class HMC {
private:
  int N; // size of grid
  Lattice q;
  Lattice p;
  double dt;
  std::random_device device;

  std::vector<double> avgQ;
  std::vector<double> avgP;

  Lattice CalculateForce(double T, double lambda, double H) {
    Lattice forceField = Lattice(N, std::vector<double>(N,0));
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        
        int ip = (i + 1) % N;
        int im = (i - 1 + N) % N;
        int jp = (j + 1) % N;
        int jm = (j - 1 + N) % N;

        forceField[i][j] = -(q[i][j] - (q[i][jp] + q[ip][j] + q[i][jm] + q[im][j]) +
            T * q[i][j] + 2 * lambda * pow(q[i][j],3) + H);
      }
    }
    return forceField;
  }

  void LeapFrogIntegrator(int L, double T, double lambda, double H) {
    Lattice forceField = CalculateForce(T, lambda, H);

    for (int k = 0; k < L; k++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          p[i][j] += 0.5 * dt * forceField[i][j];
          q[i][j] += dt * p[i][j];
        }
      }
      forceField = CalculateForce(T, lambda, H);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          p[i][j] += 0.5 * dt * forceField[i][j];
        }
      }
    }
  }

  double CalculateHamiltonian(double T, double lambda, double H) {
    double kEnergy = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        kEnergy += 0.5 * p[i][j] * p[i][j];
      }
    }

    double pEnergy = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        pEnergy += 0.5*( (q[i][j]-q[i][(j+1)%N])*(q[i][j]-q[i][(j+1)%N]) + (q[i][j]-q[(i+1)%N][j])*(q[i][j]-q[(i+1)%N][j]) + 
            T*q[i][j]*q[i][j] + lambda*q[i][j]*q[i][j]*q[i][j]*q[i][j] + H*q[i][j]);
      }
    }
    return kEnergy + pEnergy;
  }

  void UpdateLattice(int L, double T, double lambda, double H) {
    std::mt19937 generator(device());
    std::normal_distribution<double> gDistribution(0, 1);
    std::uniform_real_distribution<double> uDistribution(0, 1);

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        p[i][j] = gDistribution(generator);
      }
    }

    double initH = CalculateHamiltonian(T, lambda, H);
    Lattice qOld = q;
    LeapFrogIntegrator(L, T, lambda, H);
    double finalH = CalculateHamiltonian(T, lambda, H);

    double boltzmanProb = exp(-finalH + initH);
    double acceptionRate = (boltzmanProb > 1)? 1:boltzmanProb;

    if (uDistribution(generator)>acceptionRate) {
      q = qOld;
      return;
    }

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        p[i][j] = -p[i][j];
      }
    }
  }

  double AvgQ() {
    double result = 0; 
   
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        result += q[i][j]; 
      }
    }

    return result/(N*N);
  }

  double AvgP() {
    double result = 0; 
   
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        result += p[i][j]; 
      }
    }

    return result/(N*N);
  }

public:
  HMC(int N_p, double dt_p) : N(N_p), dt(dt_p) {
    q = Lattice(N, std::vector<double>(N,0));
    p = Lattice(N, std::vector<double>(N,0));
  }

  void InitializeLattice() {
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> uDistribution(-1, 1);

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        q[i][j] = uDistribution(generator);
      }
    }
  }

  void RunSimulation(int simSteps, int LeapSteps, double T, double lambda, double H) {
    avgQ.resize(simSteps);
    avgP.resize(simSteps);

    for (int i = 0; i < simSteps; i++) {
      UpdateLattice(LeapSteps, T, lambda, H);
      avgQ[i] = AvgQ();
      avgP[i] = AvgP();
    }
  }

  void WriteToFile(const std::string& filename, const std::vector<std::vector<double>>& phaseSpace) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return;
    }

    // Write header (optional)
    outFile << "# Step\tAvgQ\tAvgP\n";

    // Write data
    for (size_t i = 0; i < phaseSpace[0].size(); ++i) {
      outFile << i << "\t" << phaseSpace[0][i] << "\t" << phaseSpace[1][i] << "\n";
    }

    outFile.close();
    std::cout << "Data written to " << filename << std::endl;
  }

  void WriteLattice(const std::string& filename) {
    std::ofstream outFile(filename);

    outFile << "# " << q.size() << std::endl;
    for (const auto& row : q) {
        for (double val : row) {
            outFile << val << " ";
        }
        outFile << "\n";
    }
    outFile.close();
    std::cout << "Data written to " << filename << std::endl;
  }

  std::vector<double> GetAvgQ() {return avgQ;}
  std::vector<double> GetAvgP() {return avgP;}
  Lattice GetQ() {return q;}
  Lattice GetP() {return p;}
};



int main(int argc, char* argv[]) {
  int N = 16;
  double dt = 0.01;
  int simSteps = 13000;
  int LeapSteps = 10;
  double H = 0.05;
  double T = 2;
  double lambda = 0.7;

  for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-N" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "-s" && i + 1 < argc) simSteps = std::stoi(argv[++i]);
        else if (arg == "-l" && i + 1 < argc) LeapSteps = std::stoi(argv[++i]);
        else if (arg == "-T" && i + 1 < argc) T = std::stod(argv[++i]);
        else if (arg == "-lambda" && i + 1 < argc) lambda = std::stod(argv[++i]);
        else if (arg == "-H" && i + 1 < argc) H = std::stod(argv[++i]);
        else {
            std::cerr << "Unknown option or missing argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [-N grid_size] [-dt timestep] "
                      << "[-s sim_steps] [-l leap_steps] [-T temperature] "
                      << "[-lambda lambda] [-H Magnetic_field]\n";
            return 1;
        }
    }

  HMC hmc(N, dt); 
  
  hmc.InitializeLattice();

  hmc.RunSimulation(simSteps, LeapSteps, T, lambda, H);
  std::vector<std::vector<double>> phaseSpace = std::vector<std::vector<double>>(2, std::vector<double>(simSteps, 0));

  phaseSpace[0] = hmc.GetAvgQ();
  phaseSpace[1] = hmc.GetAvgP();
  hmc.WriteToFile("hmc_data.txt", phaseSpace);
  hmc.WriteLattice("hmc_mag.txt");
  
  /*
  std::ofstream out("magnetization_vs_T.txt");
  out << "#T\t<|m|>\n";

  for (double T = 0.1; T <= 3.0; T += 0.1) {
    double lambda = (T - 2.2) / T;  

    hmc.InitializeLattice();
    hmc.RunSimulation(simSteps, LeapSteps, T, lambda, H);

    
    double avg_m = 0;
    int measurement_window = 1000;
    for (int i = simSteps - measurement_window; i < simSteps; ++i) {
      avg_m += std::abs(hmc.GetAvgQ()[i]);
    }
    avg_m /= measurement_window;

    out << T << "\t" << avg_m << "\n";
    std::cout << "T = " << T << ", <|m|> = " << avg_m << std::endl;
  }

  out.close();
  */
  return 0;
}


