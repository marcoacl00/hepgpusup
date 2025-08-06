void plotData() {
  // Open the data file
  std::ifstream infile("hmc_data.txt");
  if (!infile.is_open()) {
    std::cerr << "Error opening hmc_data.txt" << std::endl;
    return;
  }

  std::vector<double> steps;
  std::vector<double> avgQ;
  std::vector<double> avgP;

  std::string line;
  // Skip header
  std::getline(infile, line);

  // Read data
  int step;
  double q, p;
  while (infile >> step >> q >> p) {
    steps.push_back(step);
    avgQ.push_back(q);
    avgP.push_back(p);
  }

  infile.close();

  int N = steps.size();

  // Create TGraph for AvgQ and AvgP
  TGraph* gQ = new TGraph(N);
  //TGraph* gP = new TGraph(N);

  for (int i = 0; i < N; ++i) {
    gQ->SetPoint(i, steps[i], avgQ[i]);
    //gP->SetPoint(i, steps[i], avgP[i]);
  }

  gQ->SetLineColor(kBlue);
  gQ->SetLineWidth(2);
  gQ->SetTitle("Average Q and P vs Step;Step;Value");

  //gP->SetLineColor(kRed);
  //gP->SetLineWidth(2);

  // Draw on canvas
  TCanvas* c = new TCanvas("c", "HMC Data", 800, 600);
  gQ->Draw("AL");
  //gP->Draw("L SAME");

  // Legend
  auto legend = new TLegend(0.7, 0.8, 0.9, 0.9);
  legend->AddEntry(gQ, "AvgQ", "l");
  //legend->AddEntry(gP, "AvgP", "l");
  legend->Draw();

  c->Update();
}
