set terminal gif animate delay 10 loop 0 size 800,600
set output "output/snapshots.gif"
set print

set title "Energy density of the jet"

set xlabel "X index"
set ylabel "Y index"

set cbrange [0:1]

set view map
set palette defined (0 "blue", 1 "red")

unset key
unset border

do for [i=0:99] {
	set label 1 sprintf("Snapshot %d", i) at graph 1, graph 1.05 center front
	splot "output/snapshots.dat" matrix index i with image
	unset label 1
	print sprintf("Snapshot: %d", i)
}

unset output
