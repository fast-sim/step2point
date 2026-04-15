#include "step2point/algorithms.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
  step2point::Shower in;
  in.shower_id = 42;
  in.x = {0.f, 2.f, 10.f};
  in.y = {0.f, 0.f, 0.f};
  in.z = {0.f, 0.f, 5.f};
  in.E = {1.f, 3.f, 2.f};
  in.cell_id = std::vector<std::uint64_t>{7, 7, 9};

  auto out = step2point::merge_within_cell(in);
  assert(out.shower.n_points() == 2);
  assert(std::abs(out.stats.energy_before - 6.0) < 1e-6);
  assert(std::abs(out.stats.energy_after - 6.0) < 1e-6);
  assert(out.shower.cell_id->at(0) == 7);
  assert(out.shower.cell_id->at(1) == 9);
  assert(std::abs(out.shower.x[0] - 1.5f) < 1e-6);
  return 0;
}
