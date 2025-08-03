g++ main.cpp -o prog -Wall -Wextra -Wpedantic
./prog | tee stdout.txt
gnuplot plot.gp
