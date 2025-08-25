#include "PlasmaModel.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
  int N = 16;
  float a = 1;
  int D = 2;
  float dt = 0.1;
  int timeSteps = 1000;
  int LeapSteps = 20;
  float lambda = 1;
  float beta = 1;
  float T = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-N" && i + 1 < argc)
      N = std::stoi(argv[++i]);
    else if (arg == "-dt" && i + 1 < argc)
      dt = std::stod(argv[++i]);
    else if (arg == "-T" && i + 1 < argc)
      T = std::stod(argv[++i]);
    else if (arg == "-s" && i + 1 < argc)
      timeSteps = std::stoi(argv[++i]);
    else if (arg == "-L" && i + 1 < argc)
      LeapSteps = std::stoi(argv[++i]);
    else if (arg == "-D" && i + 1 < argc)
      D = std::stoi(argv[++i]);
    else if (arg == "-a" && i + 1 < argc)
      a = std::stod(argv[++i]);
    else if (arg == "-lambda" && i + 1 < argc)
      lambda = std::stod(argv[++i]);
    else if (arg == "-beta" && i + 1 < argc)
      beta = std::stod(argv[++i]);
    else {
      std::cerr << "Unknown option or missing argument: " << arg << std::endl;
      return 1;
    }
  }
  PlasmaModel model(lambda, beta, N, a, D, timeSteps, dt, LeapSteps, T);
  model.InitializeGrid();
  model.RunSimulation();
  model.ExportData("data.txt");
  return 0;
}
