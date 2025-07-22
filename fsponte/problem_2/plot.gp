set terminal png
set grid

set output "plot.png"
set title "Data Visualization"
set xlabel "x"
set ylabel "y"
plot \
	"data.dat" notitle with lines lw 3
