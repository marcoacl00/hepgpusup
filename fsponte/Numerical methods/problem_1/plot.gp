set terminal png
set key outside bottom center vertical
set logscale xy
set grid

set output "plot.png"
set title "Data Visualization"
set xlabel "Step (h)"
set ylabel "Error"
plot \
	"forward_diff.dat" title "Forward Difference" with lines lw 3, \
	"central_diff.dat" title "Central Difference" with lines lw 3
