
void plotMagvsT() {
  TGraph* g = new TGraph("magnetization_vs_T.txt");
  g->SetTitle("Average Magnetization vs Temperature;T; <|m|>");
  g->SetMarkerStyle(20);
  g->Draw("APL");
}
