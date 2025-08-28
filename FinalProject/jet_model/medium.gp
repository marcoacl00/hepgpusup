set terminal png size 800,600
set output "output/medium.png"

set title "Energy density of the medium"

set xlabel "X index"
set ylabel "Y index"

set view map
set palette rgbformulae 22,13,-31

splot "output/medium.dat" matrix with image

unset output
