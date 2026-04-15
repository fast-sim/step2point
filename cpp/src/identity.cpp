#include "step2point/algorithms.hpp"

#include <stdexcept>

namespace step2point {

CompressionResult identity(const Shower& input) {
  if (!input.valid()) {
    throw std::runtime_error("Invalid Shower passed to identity().");
  }
  CompressionResult out;
  out.shower = input;
  out.algorithm = "identity";
  out.stats.n_points_before = input.n_points();
  out.stats.n_points_after = input.n_points();
  out.stats.energy_before = input.total_energy();
  out.stats.energy_after = input.total_energy();
  out.stats.compression_ratio = 1.0;
  return out;
}

} // namespace step2point
