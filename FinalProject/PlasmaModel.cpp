#include "PlasmaModel.hpp"
#include <random>

PlasmaModel::PlasmaModel(int N_p, double a_p, double dt_p): N(N_p), a(a_p), dt(dt_p), rng(device()), gDist(0.0,1.0), uDist(), q(N*N, 0), p(N*N, 0) {

}

void PlasmaModel::InitializeGrid() {
  for (int i = 0; i < N*N; i++) {
    q[i] = uDist(rng);
  }
}
