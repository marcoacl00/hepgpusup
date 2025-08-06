void plotMag(const char *filename = "hmc_mag.txt") {
  fstream infile(filename);
  if (!infile.is_open()) {
    cout << "Error: Could not open file " << filename << endl;
    return;
  }

  string line;
  getline(infile, line);
  int N = stoi(line.substr(2));

  TH2D *h =
      new TH2D("h", "HMC Lattice Field Configuration;X;Y", N, 0, N, N, 0, N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double val;
      infile >> val;
      h->SetBinContent(i + 1, j + 1, val);
    }
  }

  infile.close();

  TCanvas *c = new TCanvas("c", "", 800, 600);
  gStyle->SetOptStat(0);
  h->Draw("CONT4Z");
}
