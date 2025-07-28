set terminal png
set grid

set output "plot.png"
set title "Data Visualization"

set xlabel "Radius"
set ylabel "Potencial"

set xrange [1.9:5.1]

plot \
	"dataset.dat" using 1:2:3 with yerrorbars title "Dataset" pt 7, \
	"fit.dat" title "Fit" with lines lw 2
