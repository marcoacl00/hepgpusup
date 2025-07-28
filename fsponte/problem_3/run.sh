g++ main.cpp stat.cpp file_manager.cpp lj_potential.cpp -o prog -Wall -Wextra -Wpedantic
time -v ./prog | tee stdout.txt
gnuplot plot.gp
