#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TAxis.h"
#include "TStyle.h"
#include <Rtypes.h>
#include <RtypesCore.h>
#include <cmath>
#include <cstdlib>

#define TEST_X 0
#define MACHINE_ERROR 1E-16

Double_t TestFunction(Double_t x_p, Double_t *params) {
  return params[0] * exp(-x_p) * cos(x_p + params[1]);
}

Double_t FirstOrderErrorFunction(Double_t *x, Double_t *params) {
  Double_t h = x[0];
  Double_t A = params[0];
  Double_t phi = params[1];
  return std::abs(2 * A * std::exp(-TEST_X) * std::sin(TEST_X + phi)) * h / 2 +
         MACHINE_ERROR * std::abs(TestFunction(TEST_X, params)) / h;
}

Double_t SecondOrderErrorFunction(Double_t *x, Double_t *params) {
  Double_t h = x[0];
  Double_t A = params[0];
  Double_t phi = params[1];
  return std::abs(-4 * A * std::exp(-TEST_X) * std::cos(TEST_X + phi)) * h * h /
             12 +
         MACHINE_ERROR * std::abs(TestFunction(TEST_X, params)) / h;
}

int main(int argc, char *argv[]) {
  TApplication app("app", &argc, argv);

  Double_t params[2] = {1.0, M_PI/4};

  TF1 *firstOrderError =
      new TF1("First Order Error (forward difference)", FirstOrderErrorFunction, 1E-16, 1E-3, 2);
  TF1 *secondOrderError =
      new TF1("Second Order Error (central difference)", SecondOrderErrorFunction, 1E-16, 1E-3, 2);

  firstOrderError->SetParameters(params);
  secondOrderError->SetParameters(params);

  TCanvas *c = new TCanvas("c", "Error Functions", 800, 600);
  c->SetLogx();
  c->SetLogy();

  firstOrderError->SetLineColor(kGreen);
  secondOrderError->SetLineColor(kBlue);
  firstOrderError->SetMinimum(1E-11);
  secondOrderError->SetMinimum(1E-11);
  firstOrderError->SetTitle("Error Functions;Time Step;Error");

  firstOrderError->Draw();
  secondOrderError->Draw("SAME");

  c->BuildLegend();

  app.Run();
  return EXIT_SUCCESS;
}
