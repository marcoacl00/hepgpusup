
void plotPhaseSpace() {
  // Open the data file
  std::ifstream infile("hmc_data.txt");
  if (!infile.is_open()) {
    std::cerr << "Error: Could not open hmc_data.txt" << std::endl;
    return;
  }

  std::vector<double> avgQ;
  std::vector<double> avgP;

  std::string line;
  // Skip header
  std::getline(infile, line);

  // Read data
  int step;
  double q, p;
  while (infile >> step >> q >> p) {
    avgQ.push_back(q);
    avgP.push_back(p);
  }

  infile.close();

  int N = avgQ.size();

  // Create TGraph for phase space trajectory
  TGraph* phaseGraph = new TGraph(N);
  for (int i = 0; i < N; ++i) {
    phaseGraph->SetPoint(i, avgQ[i], avgP[i]);
  }

  phaseGraph->SetTitle("Phase Space Trajectory;AvgQ;AvgP");
  phaseGraph->SetMarkerStyle(7);  // Dots
  phaseGraph->SetMarkerColor(kBlue);
  phaseGraph->SetLineColor(kBlack);
  phaseGraph->SetLineWidth(1);

  TCanvas* c = new TCanvas("c", "Phase Space", 800, 600);
  phaseGraph->Draw("ALP");

  c->Update();
}
