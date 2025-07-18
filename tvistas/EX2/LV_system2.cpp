#include <Rtypes.h>
#include <iostream>
#include "TGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TEllipse.h"
#include "TLegend.h"
#include "TMultiGraph.h"
#include "TApplication.h"
#include "ODEsystem.hpp"


int main(int argc, char **argv) {
    TApplication app("app", &argc, argv);
    
    double preyGrowthRate = 1.1, predation1Rate = 0.4, predation2Rate = 0.3;
    double predator1Efficiency = 0.1, predator1DeathRate = 0.4, predation21Rate = 0.1;
    double predator2Efficiency = 0.07, predator2DeathRate = 0.5;
    double timeStep = 0.001;
    
    std::vector<std::function<double(double, const std::vector<double>&)>> equations = {
        [preyGrowthRate, predation1Rate, predation2Rate](double t, const std::vector<double>& y) {
            return preyGrowthRate * y[0] - predation1Rate * y[0] * y[1] - predation2Rate * y[0] * y[2];
        },
        [predator1Efficiency, predator1DeathRate, predation21Rate](double t, const std::vector<double>& y) {
            return predator1Efficiency * y[0] * y[1] - predator1DeathRate * y[1] - predation21Rate * y[1] * y[2]; 
        },
        [predator2Efficiency, predator2DeathRate](double t, const std::vector<double>& y) {
            return predator2Efficiency * y[0] * y[2] + predator2Efficiency * y[1] * y[2] -  predator2DeathRate * y[2];
        }
    };
    
    std::vector<double> initialPopulations = {15.0, 7.0, 3.0};
    
    ODEsystem lotkaVolterraSystem(equations, initialPopulations, 0.0, 50.0, timeStep);
    auto populationResults = lotkaVolterraSystem.solveSystem();
    
    int numSteps = populationResults[0].size();
    std::vector<double> timeValues(numSteps);
    for (int i = 0; i < numSteps; i++) {
        timeValues[i] = 0.0 + i * timeStep;
    }
    
    TCanvas *canvas = new TCanvas("populationCanvas", "Population Dynamics", 800, 600);
    TGraph *preyPopulationGraph = new TGraph(numSteps, &timeValues[0], &populationResults[0][0]);
    TGraph *predator1PopulationGraph = new TGraph(numSteps, &timeValues[0], &populationResults[1][0]);
    TGraph *predator2PopulationGraph = new TGraph(numSteps, &timeValues[0], &populationResults[2][0]);
    
    preyPopulationGraph->SetTitle("Lotka-Volterra Model;Time;Population");
    preyPopulationGraph->SetLineColor(kBlue);
    preyPopulationGraph->SetLineWidth(2);
    predator1PopulationGraph->SetLineColor(kRed);
    predator1PopulationGraph->SetLineWidth(2);
    predator2PopulationGraph->SetLineColor(kGreen);
    predator2PopulationGraph->SetLineWidth(2);
    
    TMultiGraph *populationMultiGraph = new TMultiGraph();
    populationMultiGraph->Add(preyPopulationGraph);
    populationMultiGraph->Add(predator1PopulationGraph);
    populationMultiGraph->Add(predator2PopulationGraph);
    populationMultiGraph->Draw("AL");
    
    TLegend *populationLegend = new TLegend(0.7, 0.7, 0.9, 0.9);
    populationLegend->AddEntry(preyPopulationGraph, "Prey", "l");
    populationLegend->AddEntry(predator1PopulationGraph, "Predator1", "l");
    populationLegend->AddEntry(predator2PopulationGraph, "Predator2", "l");
    populationLegend->Draw();
    
    std::cout << "Equilibrium Points Analysis:\n";
    std::cout << "1. Extinction (0, 0) - Saddle point (unstable)\n";
    
    app.Run();
    return 0;
}
