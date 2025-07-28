void plot_data() {
    // Create a ROOT graph from the data file
    TGraph *graph = new TGraph("data.txt");
    
    // Customize the graph
    graph->SetTitle("Position Expectation Value vs Time");
    graph->GetXaxis()->SetTitle("Time (t)");
    graph->GetYaxis()->SetTitle("<x>(t)");
    graph->SetLineColor(kBlue);
    graph->SetLineWidth(2);
    
    // Draw the graph
    TCanvas *c1 = new TCanvas("c1", "Position Expectation", 800, 600);
    graph->Draw("AL");
    
    // Save the plot to a file
    c1->SaveAs("position_expectation.png");
}
