#include "step2point/shower.hpp"

#include <numeric>

namespace step2point {

double Shower::total_energy() const noexcept {
  return std::accumulate(E.begin(), E.end(), 0.0);
}

bool Shower::valid() const noexcept {
  const auto n = E.size();
  if (x.size() != n || y.size() != n || z.size() != n) return false;
  if (t && t->size() != n) return false;
  if (cell_id && cell_id->size() != n) return false;
  return true;
}

} // namespace step2point
