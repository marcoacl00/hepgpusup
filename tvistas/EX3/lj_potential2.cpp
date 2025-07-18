#include "ODEsystem.hpp"
#include "TApplication.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

double lennard_jones_potential(double r, double eps = 1.0, double sigma = 1.0) {
    double sr6 = pow(sigma/r, 6);
    double sr12 = sr6 * sr6;
    return 4.0 * eps * (sr12 - sr6);
}

double HO_potential(double r, double k = 1.0) {
  return 0.5*k*r*r;
}

int main(int argc, char *argv[]) {
    TApplication app("app", &argc, argv);

    const double timeStep = 0.01;
    const double eps = 1.0;
    const double sigma = 1.0;
    const double r_eq = sigma * pow(2.0, 1.0/6.0);

    std::vector<std::function<double(double, const std::vector<double>&)>> equations = {
        [](double t, const std::vector<double>& y) { return y[1]; },
        [eps, sigma](double t, const std::vector<double>& y) {
            double r = y[0];
            //return 4.0*eps*(12.0*pow(sigma,12)/pow(r,13) - 6.0*pow(sigma,6)/pow(r,7));
            return -r;
        }
    };

    std::vector<double> initConditions = {1, 0.0};
    ODEsystem system(equations, initConditions, 0.0, 10.0, timeStep);
    auto solutions = system.solveSystem();

    int numSteps = solutions[0].size();
    std::vector<double> timeValues(numSteps);
    std::vector<double> energyValues(numSteps);
    std::vector<double> potentialValues(numSteps);
    std::vector<double> kineticValues(numSteps);

    for (int i = 0; i < numSteps; i++) {
        timeValues[i] = i * timeStep;
        double r = solutions[0][i];
        double v = solutions[1][i];
        potentialValues[i] = HO_potential(r);
        kineticValues[i] = 0.5 * v * v;  // μ=1 so KE = 0.5μv² → 0.5v²
        energyValues[i] = kineticValues[i] + potentialValues[i];
    }

    TCanvas *canvas = new TCanvas("canvas", "Energy Analysis", 1200, 800);
    canvas->Divide(2,1);
    
    canvas->cd(1);
    TGraph *pos_graph = new TGraph(numSteps, timeValues.data(), solutions[0].data());
    pos_graph->SetTitle("Position;Time;r(t)");
    pos_graph->SetLineColor(kBlue);
    pos_graph->Draw("AL");

    canvas->cd(2);
    TMultiGraph *energy_plot = new TMultiGraph();
    
    TGraph *total_energy = new TGraph(numSteps, timeValues.data(), energyValues.data());
    total_energy->SetLineColor(kBlack);
    total_energy->SetLineWidth(2);
    
    TGraph *potential = new TGraph(numSteps, timeValues.data(), potentialValues.data());
    potential->SetLineColor(kRed);
    
    TGraph *kinetic = new TGraph(numSteps, timeValues.data(), kineticValues.data());
    kinetic->SetLineColor(kGreen);
    
    energy_plot->Add(total_energy);
    energy_plot->Add(potential);
    energy_plot->Add(kinetic);
    energy_plot->SetTitle("Energy Components;Time;Energy");
    energy_plot->Draw("AL");

    TLegend *leg = new TLegend(0.7,0.7,0.9,0.9);
    leg->AddEntry(total_energy,"Total Energy","l");
    leg->AddEntry(potential,"Potential","l");
    leg->AddEntry(kinetic,"Kinetic","l");
    leg->Draw();
    
    double energy_error = *max_element(energyValues.begin(),energyValues.end()) 
      - *min_element(energyValues.begin(),energyValues.end());
    std::cout << "Max energy fluctuation: " << energy_error << std::endl;
    app.Run();

    return 0;
}
