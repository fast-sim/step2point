#include "step2point/algorithms.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace step2point {

CompressionResult merge_within_cell(const Shower& input) {
  if (!input.valid()) {
    throw std::runtime_error("Invalid Shower passed to merge_within_cell().");
  }
  if (!input.cell_id) {
    throw std::runtime_error("merge_within_cell() requires cell_id.");
  }

  struct Acc {
    double e = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double t = 0.0;
  };

  std::unordered_map<std::uint64_t, Acc> accum;
  accum.reserve(input.n_points());
  std::vector<std::uint64_t> cell_order;
  cell_order.reserve(input.n_points());

  for (std::size_t i = 0; i < input.n_points(); ++i) {
    const auto cell = (*input.cell_id)[i];
    auto [it, inserted] = accum.try_emplace(cell, Acc{});
    if (inserted) {
      cell_order.push_back(cell);
    }
    auto& a = it->second;
    const double e = input.E[i];
    a.e += e;
    a.x += input.x[i] * e;
    a.y += input.y[i] * e;
    a.z += input.z[i] * e;
    if (input.t) a.t += (*input.t)[i] * e;
  }

  std::sort(cell_order.begin(), cell_order.end());

  Shower out;
  out.shower_id = input.shower_id;
  out.cell_id = std::vector<std::uint64_t>{};
  if (input.t) out.t = std::vector<float>{};

  out.x.reserve(cell_order.size());
  out.y.reserve(cell_order.size());
  out.z.reserve(cell_order.size());
  out.E.reserve(cell_order.size());
  out.cell_id->reserve(cell_order.size());
  if (out.t) out.t->reserve(cell_order.size());

  for (const auto cell : cell_order) {
    const auto& a = accum.at(cell);
    const double safe = a.e > 0.0 ? a.e : 1.0;
    out.cell_id->push_back(cell);
    out.x.push_back(static_cast<float>(a.x / safe));
    out.y.push_back(static_cast<float>(a.y / safe));
    out.z.push_back(static_cast<float>(a.z / safe));
    out.E.push_back(static_cast<float>(a.e));
    if (out.t) out.t->push_back(static_cast<float>(a.t / safe));
  }

  CompressionResult result;
  result.shower = std::move(out);
  result.algorithm = "merge_within_cell";
  result.stats.n_points_before = input.n_points();
  result.stats.n_points_after = result.shower.n_points();
  result.stats.energy_before = input.total_energy();
  result.stats.energy_after = result.shower.total_energy();
  result.stats.compression_ratio = input.n_points() == 0
      ? 1.0
      : static_cast<double>(result.shower.n_points()) / static_cast<double>(input.n_points());
  return result;
}

} // namespace step2point
