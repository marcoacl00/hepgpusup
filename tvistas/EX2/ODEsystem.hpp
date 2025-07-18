#ifndef ODESYSTEM_H
#define ODESYSTEM_H

#include <functional>
#include <vector>


class ODEsystem {

private:
  std::vector<std::function<double(double, const std::vector<double> &)>>
      odeEquations;
  std::vector<double> initialConditions;
  double initialTime, finalTime, stepSize;

public:
  ODEsystem(const std::vector<std::function<double(double, const std::vector<double>&)>>& equations,
              const std::vector<double>& initConditions,
              double startTime = 0.0,
              double endTime = 10.0,
              double timeStep = 0.01)
        : odeEquations(equations), initialConditions(initConditions),
          initialTime(startTime), finalTime(endTime), stepSize(timeStep) {}

  std::vector<std::vector<double>> solveSystem() {
    int numEquations = odeEquations.size();
    int numSteps = static_cast<int>((finalTime - initialTime) / stepSize);

    std::vector<std::vector<double>> solutions(numEquations,
                                               std::vector<double>(numSteps));
    std::vector<double> currentValues = initialConditions;

    for (int i = 0; i < numEquations; i++) {
      solutions[i][0] = initialConditions[i];
    }

    for (int step = 1; step < numSteps; step++) {
      double currentTime = initialTime + step * stepSize;
      std::vector<double> nextValues(numEquations);

      std::vector<double> k1(numEquations), k2(numEquations), k3(numEquations),
          k4(numEquations);
      std::vector<double> tempValues(numEquations);

      //  k1
      for (int i = 0; i < numEquations; i++) {
        k1[i] = odeEquations[i](currentTime, currentValues);
        tempValues[i] = currentValues[i] + 0.5 * stepSize * k1[i];
      }

      //  k2
      for (int i = 0; i < numEquations; i++) {
        k2[i] = odeEquations[i](currentTime + 0.5 * stepSize, tempValues);
        tempValues[i] = currentValues[i] + 0.5 * stepSize * k2[i];
      }

      //  k3
      for (int i = 0; i < numEquations; i++) {
        k3[i] = odeEquations[i](currentTime + 0.5 * stepSize, tempValues);
        tempValues[i] = currentValues[i] + stepSize * k3[i];
      }

      //  k4
      for (int i = 0; i < numEquations; i++) {
        k4[i] = odeEquations[i](currentTime + stepSize, tempValues);
        nextValues[i] =
            currentValues[i] +
            stepSize * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
      }

      for (int i = 0; i < numEquations; i++) {
        solutions[i][step] = nextValues[i];
      }
      currentValues = nextValues;
    }

    return solutions;
  }
};

#endif // !ODESYSTEM_H
