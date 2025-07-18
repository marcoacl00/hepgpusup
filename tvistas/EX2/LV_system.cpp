#include <iostream>
#include <vector>
#include <functional>
#include "TGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TEllipse.h"
#include "TLegend.h"
#include "TMultiGraph.h"
#include "TApplication.h"

class OdeSystem {
private:
    std::vector<std::function<double(double, const std::vector<double>&)>> odeEquations;
    std::vector<double> initialConditions;
    double initialTime, finalTime, stepSize;
    
public:
    OdeSystem(const std::vector<std::function<double(double, const std::vector<double>&)>>& equations,
              const std::vector<double>& initConditions,
              double startTime = 0.0,
              double endTime = 10.0,
              double timeStep = 0.01)
        : odeEquations(equations), initialConditions(initConditions),
          initialTime(startTime), finalTime(endTime), stepSize(timeStep) {}
    
    std::vector<std::vector<double>> solveSystem() {
        int numEquations = odeEquations.size();
        int numSteps = static_cast<int>((finalTime - initialTime) / stepSize);
        
        std::vector<std::vector<double>> solutions(numEquations, std::vector<double>(numSteps));
        std::vector<double> currentValues = initialConditions;
        
        for (int i = 0; i < numEquations; i++) {
            solutions[i][0] = initialConditions[i];
        }
        
        for (int step = 1; step < numSteps; step++) {
            double currentTime = initialTime + step * stepSize;
            std::vector<double> nextValues(numEquations);
            
            std::vector<double> k1(numEquations), k2(numEquations), k3(numEquations), k4(numEquations);
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
                nextValues[i] = currentValues[i] + stepSize * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
            }
            
            for (int i = 0; i < numEquations; i++) {
                solutions[i][step] = nextValues[i];
            }
            currentValues = nextValues;
        }
        
        return solutions;
    }
};

int main(int argc, char **argv) {
    TApplication app("app", &argc, argv);
    
    double preyGrowthRate = 1.1, predationRate = 0.4;
    double predatorEfficiency = 0.1, predatorDeathRate = 0.4;
    double timeStep = 0.001;
    
    std::vector<std::function<double(double, const std::vector<double>&)>> equations = {
        [preyGrowthRate, predationRate](double t, const std::vector<double>& y) {
            return preyGrowthRate * y[0] - predationRate * y[0] * y[1];
        },
        [predatorEfficiency, predatorDeathRate](double t, const std::vector<double>& y) {
            return predatorEfficiency * y[0] * y[1] - predatorDeathRate * y[1]; 
        }
    };
    
    std::vector<double> initialPopulations = {10.0, 5.0};
    
    OdeSystem lotkaVolterraSystem(equations, initialPopulations, 0.0, 50.0, timeStep);
    auto populationResults = lotkaVolterraSystem.solveSystem();
    
    int numSteps = populationResults[0].size();
    std::vector<double> timeValues(numSteps);
    for (int i = 0; i < numSteps; i++) {
        timeValues[i] = 0.0 + i * timeStep;
    }
    
    TCanvas *canvas = new TCanvas("populationCanvas", "Population Dynamics", 800, 600);
    TGraph *preyPopulationGraph = new TGraph(numSteps, &timeValues[0], &populationResults[0][0]);
    TGraph *predatorPopulationGraph = new TGraph(numSteps, &timeValues[0], &populationResults[1][0]);
    
    preyPopulationGraph->SetTitle("Lotka-Volterra Model;Time;Population");
    preyPopulationGraph->SetLineColor(kBlue);
    preyPopulationGraph->SetLineWidth(2);
    predatorPopulationGraph->SetLineColor(kRed);
    predatorPopulationGraph->SetLineWidth(2);
    
    TMultiGraph *populationMultiGraph = new TMultiGraph();
    populationMultiGraph->Add(preyPopulationGraph);
    populationMultiGraph->Add(predatorPopulationGraph);
    populationMultiGraph->Draw("AL");
    
    TLegend *populationLegend = new TLegend(0.7, 0.7, 0.9, 0.9);
    populationLegend->AddEntry(preyPopulationGraph, "Prey", "l");
    populationLegend->AddEntry(predatorPopulationGraph, "Predator", "l");
    populationLegend->Draw();
    
    TCanvas *phaseSpaceCanvas = new TCanvas("phaseSpaceCanvas", "Phase Space", 800, 600);
    TGraph *phaseSpaceGraph = new TGraph(numSteps, &populationResults[0][0], &populationResults[1][0]);
    phaseSpaceGraph->SetTitle("Phase Space;Prey Population;Predator Population");
    phaseSpaceGraph->SetLineColor(kGreen+2);
    phaseSpaceGraph->SetLineWidth(2);
    phaseSpaceGraph->Draw("AL");
    
    double preyEquilibrium = predatorDeathRate / predatorEfficiency;
    double predatorEquilibrium = preyGrowthRate / predationRate;
    
    TEllipse *equilibriumPoint = new TEllipse(preyEquilibrium, predatorEquilibrium, 0.5, 0.5);
    equilibriumPoint->SetFillColor(kRed);
    equilibriumPoint->SetLineColor(kRed);
    equilibriumPoint->Draw("same");
    
    std::cout << "Equilibrium Points Analysis:\n";
    std::cout << "1. Extinction (0, 0) - Saddle point (unstable)\n";
    std::cout << "2. Coexistence (" << preyEquilibrium << ", " << predatorEquilibrium << ") - Center (neutrally stable)\n";
    
    app.Run();
    return 0;
}
